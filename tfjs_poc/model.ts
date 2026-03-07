/**
 * TF.js implementation of TransformerActorCritic for inference.
 *
 * Matches the JAX/Flax model in jaxrl/model/ exactly.
 * All tensor shapes are annotated as comments.
 */

import * as tf from "@tensorflow/tfjs";

// ──────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────

export interface ModelConfig {
  hidden_features: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  head_dim: number;
  rope_max_wavelength: number;
  ffn_size: number;
  action_dim: number;
  max_seq_length: number;
  obs_shape: number[];
  obs_max_value: number[];
  obs_one_hot_total: number;
  conv_kernels: number[][];
  conv_strides: number[][];
  conv_channels: number[];
  value_hidden_dim: number | null;
  value_n_logits: number;
}

export interface KVCache {
  key: tf.Tensor;   // [batch, maxSeqLength, numKvHeads, headDim]
  value: tf.Tensor; // [batch, maxSeqLength, numKvHeads, headDim]
}

export interface Timestep {
  obs: tf.Tensor;        // [batch, H, W, C] int32
  reward: tf.Tensor;     // [batch]
  lastAction: tf.Tensor; // [batch] int32
  time: tf.Tensor;       // [batch] int32
}

export interface ForwardResult {
  actionProbs: tf.Tensor;   // [batch, actionDim]
  actionLogits: tf.Tensor;  // [batch, actionDim]
  kvCaches: KVCache[];
}

interface AttentionWeights {
  query_proj: tf.Tensor; // [d_model, numHeads, headDim]
  key_proj: tf.Tensor;   // [d_model, numKvHeads, headDim]
  value_proj: tf.Tensor; // [d_model, numKvHeads, headDim]
  out: tf.Tensor;        // [numHeads, headDim, d_model]
}

interface FFNWeights {
  up_proj: tf.Tensor;   // [d_model, ffnSize]
  up_gate: tf.Tensor;   // [d_model, ffnSize]
  down_proj: tf.Tensor; // [ffnSize, d_model]
}

interface AttentionConfig {
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  ropeMaxWavelength: number;
  maxSeqLength: number;
}

interface AttentionResult {
  out: tf.Tensor;
  kvCache: KVCache;
}

/** Weight dict: name → tensor, plus conv strides from config. */
export type Weights = Record<string, tf.Tensor> & { _convStrides: number[][] };

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

/**
 * Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
 * Matches jax.nn.gelu (which uses exact by default).
 */
function gelu(x: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const cdf = tf.mul(
      0.5,
      tf.add(1.0, tf.erf(tf.mul(x, Math.SQRT1_2)))
    );
    return tf.mul(x, cdf);
  });
}

/** RMSNorm: x * rsqrt(mean(x^2) + eps) * scale */
function rmsNorm(x: tf.Tensor, scale: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const ms = tf.mean(tf.square(x), -1, /* keepDims */ true);
    const rms = tf.rsqrt(tf.add(ms, 1e-6));
    return tf.mul(tf.mul(x, rms), scale);
  });
}

/**
 * concat_one_hot — matches mapox.concat_one_hot exactly.
 *
 * Input x has shape [..., n] where each of the n components has its own
 * vocabulary size given by `sizes`.  Output has shape [..., sum(sizes)]
 * with a 1.0 at the offset position for each component.
 */
function concatOneHot(x: tf.Tensor, sizes: number[]): tf.Tensor {
  return tf.tidy(() => {
    const total = sizes.reduce((a, b) => a + b, 0);
    const batchShape = x.shape.slice(0, -1);
    const n = x.shape[x.shape.length - 1];
    const flatBatch = batchShape.reduce((a, b) => a * b, 1);

    // Cumulative offsets: [0, sizes[0], sizes[0]+sizes[1], ...]
    const offsets = new Int32Array(n);
    for (let i = 1; i < n; i++) offsets[i] = offsets[i - 1] + sizes[i - 1];
    const offsetsTensor = tf.tensor1d(offsets, "int32");

    const flat = tf.reshape(x, [flatBatch, n]);
    const idx = tf.add(tf.cast(flat, "int32"), offsetsTensor); // [flatBatch, n]

    // Scatter ones
    const out = tf.buffer([flatBatch, total], "float32");
    const idxData = idx.dataSync();
    for (let b = 0; b < flatBatch; b++) {
      for (let c = 0; c < n; c++) {
        out.set(1, b, idxData[b * n + c]);
      }
    }

    return tf.reshape(out.toTensor(), [...batchShape, total]);
  });
}

// ──────────────────────────────────────────────
// Convolution encoder
// ──────────────────────────────────────────────

/**
 * 3-layer CNN observation encoder (GridCnnObsEncoder).
 *
 * Applies one-hot encoding, then 3 conv2d layers with VALID padding.
 * GELU activation between layers but NOT after the last layer.
 */
function gridCnnEncoder(
  obs: tf.Tensor,
  weights: Weights,
  oneHotSizes: number[],
): tf.Tensor {
  return tf.tidy(() => {
    const [batch, seq, H, W, C] = obs.shape;

    // Merge batch and seq dims for conv2d (which expects 4D)
    let x: tf.Tensor = tf.reshape(obs, [batch * seq, H, W, C]);

    // One-hot encode per spatial position: [bs*H*W, C] → [bs*H*W, total]
    x = tf.reshape(x, [batch * seq * H * W, C]);
    x = concatOneHot(x, oneHotSizes);
    x = tf.reshape(x, [batch * seq, H, W, -1]); // [bs, 11, 11, 30]

    // Conv layers
    for (let i = 0; i < 3; i++) {
      const kernel = weights[`obs_encoder.layers.${i}.kernel`] as tf.Tensor4D;
      const bias = weights[`obs_encoder.layers.${i}.bias`];
      // Flax Conv kernel: [kH, kW, Cin, Cout] — same as TF.js
      x = tf.conv2d(x as tf.Tensor4D, kernel, weights._convStrides[i] as [number, number], "valid");
      x = tf.add(x, bias);
      if (i < 2) {
        x = gelu(x);
      }
    }

    // Flatten spatial dims: [bs, 1, 1, 128] → [bs, 128]
    const outFeatures = x.shape[x.shape.length - 1];
    return tf.reshape(x, [batch, seq, outFeatures]);
  });
}

// ──────────────────────────────────────────────
// RoPE (Rotary Position Embeddings)
// ──────────────────────────────────────────────

/** Apply RoPE to query or key tensor: [batch, seq, numHeads, headDim]. */
function applyRope(
  inputs: tf.Tensor,
  positions: tf.Tensor,
  headDim: number,
  maxWavelength: number,
): tf.Tensor {
  return tf.tidy(() => {
    const halfDim = headDim / 2;

    // Frequency basis: max_wavelength^(2i/headDim)
    const fraction = tf.div(
      tf.mul(tf.range(0, halfDim, 1, "float32"), 2),
      headDim
    );
    const timescale = tf.pow(tf.scalar(maxWavelength), fraction); // [halfDim]

    // positions: [batch, seq] → [batch, seq, 1]
    const posExpanded = tf.expandDims(tf.cast(positions, "float32"), -1);
    const tsExpanded = tf.reshape(timescale, [1, 1, halfDim]);
    let sinusoidInp = tf.div(posExpanded, tsExpanded); // [batch, seq, halfDim]

    // Add head dimension: [batch, seq, 1, halfDim]
    sinusoidInp = tf.expandDims(sinusoidInp, 2);

    const sinVal = tf.sin(sinusoidInp);
    const cosVal = tf.cos(sinusoidInp);

    const [firstHalf, secondHalf] = tf.split(inputs, 2, -1);

    const firstPart = tf.sub(tf.mul(firstHalf, cosVal), tf.mul(secondHalf, sinVal));
    const secondPart = tf.add(tf.mul(secondHalf, cosVal), tf.mul(firstHalf, sinVal));

    return tf.concat([firstPart, secondPart], -1);
  });
}

// ──────────────────────────────────────────────
// Grouped Query Attention with KV cache
// ──────────────────────────────────────────────

/**
 * Update a KV cache tensor at a specific sequence position.
 * cache: [batch, maxSeq, heads, headDim]
 * update: [batch, 1, heads, headDim]
 */
function updateCache(cache: tf.Tensor, update: tf.Tensor, pos: number): tf.Tensor {
  return tf.tidy(() => {
    const [, maxSeq] = cache.shape;

    // One-hot mask for the position: [1, maxSeq, 1, 1]
    const mask = tf.oneHot(tf.scalar(pos, "int32"), maxSeq)
      .reshape([1, maxSeq, 1, 1]);

    // new_cache = cache * (1 - mask) + tiled_update * mask
    const inverseMask = tf.sub(tf.scalar(1), mask);
    const updateTiled = tf.mul(
      tf.tile(update, [1, maxSeq, 1, 1]),
      mask
    );

    return tf.add(tf.mul(cache, inverseMask), updateTiled);
  });
}

/**
 * Single-step GQA with KV cache update.
 * For inference we always process seq=1 with a rolling KV cache.
 */
function gqaAttention(
  x: tf.Tensor,
  seqPos: tf.Tensor,
  layerW: AttentionWeights,
  kvCache: KVCache,
  cfg: AttentionConfig,
): AttentionResult {
  // Can't use tf.tidy for the whole body since we return non-tensor KVCache.
  // Instead, manage tensors manually via tf.keep on outputs.
  return tf.tidy(() => {
    const batch = x.shape[0]!;

    // Project to Q, K, V
    let query = tf.einsum("bsd,dhk->bshk", x, layerW.query_proj); // [b,1,numHeads,headDim]
    let key = tf.einsum("bsd,dhk->bshk", x, layerW.key_proj);     // [b,1,kvHeads,headDim]
    const value = tf.einsum("bsd,dhk->bshk", x, layerW.value_proj); // [b,1,kvHeads,headDim]

    // Apply RoPE to Q and K
    query = applyRope(query, seqPos, cfg.headDim, cfg.ropeMaxWavelength);
    key = applyRope(key, seqPos, cfg.headDim, cfg.ropeMaxWavelength);

    // Update KV cache at pos % maxSeqLength
    const pos = seqPos.dataSync()[0] % cfg.maxSeqLength;
    const newKeyCache = updateCache(kvCache.key, key, pos);
    const newValueCache = updateCache(kvCache.value, value, pos);

    // Determine how many KV entries are valid
    const kvLength = Math.min(seqPos.dataSync()[0] + 1, cfg.maxSeqLength);

    // Slice valid KV entries: [batch, kvLength, numKvHeads, headDim]
    const validKeys = tf.slice(newKeyCache, [0, 0, 0, 0], [batch, kvLength, cfg.numKvHeads, cfg.headDim]);
    const validValues = tf.slice(newValueCache, [0, 0, 0, 0], [batch, kvLength, cfg.numKvHeads, cfg.headDim]);

    // Scaled dot-product attention with GQA via broadcasting.
    // Q:  [b, numHeads,   1,     headDim]
    // K:  [b, numKvHeads, headDim, kvLen]  → broadcasts over numHeads
    const scale = 1.0 / Math.sqrt(cfg.headDim);

    const qT = tf.transpose(query, [0, 2, 1, 3]);       // [b, numHeads, 1, headDim]
    const kT = tf.transpose(validKeys, [0, 2, 3, 1]);   // [b, numKvHeads, headDim, kvLen]
    let scores = tf.matMul(qT, kT);                      // [b, numHeads, 1, kvLen]
    scores = tf.mul(scores, scale);

    const attnWeights = tf.softmax(scores, -1);

    const vT = tf.transpose(validValues, [0, 2, 1, 3]); // [b, numKvHeads, kvLen, headDim]
    let attnOut = tf.matMul(attnWeights, vT);            // [b, numHeads, 1, headDim]

    // Transpose back: [b, 1, numHeads, headDim]
    attnOut = tf.transpose(attnOut, [0, 2, 1, 3]);

    // Output projection: [b, 1, numHeads, headDim] → [b, 1, d_model]
    const out = tf.einsum("bshk,hkd->bsd", attnOut, layerW.out);

    return { out, kvCache: { key: newKeyCache, value: newValueCache } };
  }) as unknown as AttentionResult;
}

// ──────────────────────────────────────────────
// GLU Feed-Forward
// ──────────────────────────────────────────────

/** GLU FFN: gelu(x @ up_proj) * (x @ up_gate) then @ down_proj */
function gluFeedForward(x: tf.Tensor, layerW: FFNWeights): tf.Tensor {
  return tf.tidy(() => {
    const up = tf.matMul(x, layerW.up_proj);
    const gate = tf.matMul(x, layerW.up_gate);
    const geluUp = gelu(up);
    const gated = tf.mul(geluUp, gate);
    return tf.matMul(gated, layerW.down_proj);
  });
}

// ──────────────────────────────────────────────
// Embedding
// ──────────────────────────────────────────────

/** Embedding lookup with sqrt scaling (matches Embedder.encode). */
function embedEncode(indices: tf.Tensor, table: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const emb = tf.gather(table, tf.cast(indices, "int32"));
    const scale = Math.sqrt(table.shape[1]!);
    return tf.mul(emb, scale);
  });
}

/** Embedding decode: x @ table^T (project back to vocab space). */
function embedDecode(x: tf.Tensor, table: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    return tf.matMul(x, tf.transpose(table));
  });
}

// ──────────────────────────────────────────────
// Full forward pass
// ──────────────────────────────────────────────

/** Initialize KV caches for all layers. */
export function initializeKvCaches(batchSize: number, cfg: ModelConfig): KVCache[] {
  const caches: KVCache[] = [];
  for (let i = 0; i < cfg.num_layers; i++) {
    caches.push({
      key: tf.zeros([batchSize, cfg.max_seq_length, cfg.num_kv_heads, cfg.head_dim]),
      value: tf.zeros([batchSize, cfg.max_seq_length, cfg.num_kv_heads, cfg.head_dim]),
    });
  }
  return caches;
}

/** Run a single-step forward pass through the TransformerActorCritic. */
export function forward(
  timestep: Timestep,
  kvCaches: KVCache[],
  weights: Weights,
  cfg: ModelConfig,
): ForwardResult {
  const newCaches: KVCache[] = [];

  const result = tf.tidy(() => {
    // Add sequence dimension: [batch, ...] → [batch, 1, ...]
    const obs = tf.expandDims(timestep.obs, 1);
    const reward = tf.expandDims(timestep.reward, 1);
    const lastAction = tf.expandDims(timestep.lastAction, 1);
    const time = tf.expandDims(timestep.time, 1);

    // ── Observation encoding ──
    const obsEmb = gridCnnEncoder(obs, weights, cfg.obs_max_value);

    // ── Reward encoding ──
    const rewardExpanded = tf.expandDims(reward, -1); // [b, 1, 1]
    const rewardEmb = tf.add(
      tf.matMul(rewardExpanded, weights["reward_encoder.kernel"]),
      weights["reward_encoder.bias"]
    );

    // ── Action embedding ──
    const actionEmb = embedEncode(lastAction, weights["action_embedder.embedding_table"]);

    // ── Combine ──
    let x = tf.add(tf.add(obsEmb, rewardEmb), actionEmb);

    // ── Transformer layers ──
    for (let i = 0; i < cfg.num_layers; i++) {
      const prefix = `layers.${i}`;

      // Pre-attention RMSNorm
      const histNormed = rmsNorm(x, weights[`${prefix}.history_norm.scale`]);

      // GQA Attention
      const layerAttnW: AttentionWeights = {
        query_proj: weights[`${prefix}.history.query_proj.kernel`],
        key_proj: weights[`${prefix}.history.key_proj.kernel`],
        value_proj: weights[`${prefix}.history.value_proj.kernel`],
        out: weights[`${prefix}.history.out.kernel`],
      };

      const attnCfg: AttentionConfig = {
        numHeads: cfg.num_heads,
        numKvHeads: cfg.num_kv_heads,
        headDim: cfg.head_dim,
        ropeMaxWavelength: cfg.rope_max_wavelength,
        maxSeqLength: cfg.max_seq_length,
      };

      const attnResult = gqaAttention(histNormed, time, layerAttnW, kvCaches[i], attnCfg);

      newCaches.push({
        key: tf.keep(attnResult.kvCache.key),
        value: tf.keep(attnResult.kvCache.value),
      });

      // Residual
      x = tf.add(x, attnResult.out);

      // Pre-FFN RMSNorm
      const ffnNormed = rmsNorm(x, weights[`${prefix}.ffn_norm.scale`]);

      // GLU FFN
      const ffnW: FFNWeights = {
        up_proj: weights[`${prefix}.ffn.up_proj.kernel`],
        up_gate: weights[`${prefix}.ffn.up_gate.kernel`],
        down_proj: weights[`${prefix}.ffn.down_proj.kernel`],
      };
      const ffnOut = gluFeedForward(ffnNormed, ffnW);

      // Residual
      x = tf.add(x, ffnOut);
    }

    // ── Output norm ──
    x = rmsNorm(x, weights["output_norm.scale"]);

    // ── Action logits via embedding decode ──
    const actionLogits = embedDecode(x, weights["action_embedder.embedding_table"]);

    // Remove seq dim and compute softmax
    const logitsFlat = tf.squeeze(actionLogits, [1]);
    const actionProbs = tf.softmax(logitsFlat, -1);

    return { actionProbs: tf.keep(actionProbs), actionLogits: tf.keep(logitsFlat) };
  });

  // Dispose old caches
  for (const cache of kvCaches) {
    cache.key.dispose();
    cache.value.dispose();
  }

  return {
    actionProbs: result.actionProbs,
    actionLogits: result.actionLogits,
    kvCaches: newCaches,
  };
}

// ──────────────────────────────────────────────
// Weight loading
// ──────────────────────────────────────────────

interface WeightsManifest {
  weightsManifest: Array<{
    paths: string[];
    weights: Array<{ name: string; shape: number[]; dtype: string }>;
  }>;
}

/** Load weights from TF.js model.json manifest. */
export async function loadWeights(modelJsonUrl: string, cfg: ModelConfig): Promise<Weights> {
  const baseUrl = modelJsonUrl.replace(/\/[^/]*$/, "/");

  const resp = await fetch(modelJsonUrl);
  const manifest: WeightsManifest = await resp.json();

  const group = manifest.weightsManifest[0];
  const shardUrl = baseUrl + group.paths[0];
  const shardResp = await fetch(shardUrl);
  const shardBuffer = await shardResp.arrayBuffer();

  const weights: Record<string, tf.Tensor> = {};
  let offset = 0;

  for (const spec of group.weights) {
    const numElements = spec.shape.reduce((a, b) => a * b, 1);
    const byteLength = numElements * 4; // float32
    const data = new Float32Array(shardBuffer, offset, numElements);
    weights[spec.name] = tf.tensor(data, spec.shape, "float32");
    offset += byteLength;
  }

  return Object.assign(weights, { _convStrides: cfg.conv_strides }) as Weights;
}
