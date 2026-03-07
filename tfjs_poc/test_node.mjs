/**
 * Node.js test for the TF.js model implementation.
 * Loads weights from disk (no HTTP server needed) and verifies against reference.
 *
 * Usage: node tfjs_poc/test_node.mjs
 */

import * as tf from "@tensorflow/tfjs";
import { readFileSync } from "fs";
import { forward, loadWeights, initializeKvCaches } from "./dist/model.js";

// Load config
const cfg = JSON.parse(readFileSync("tfjs_poc/weights/config.json", "utf-8"));
console.log(`Config: ${cfg.num_layers} layers, ${cfg.hidden_features}d, ${cfg.num_heads} heads`);

// Load reference
const ref = JSON.parse(readFileSync("tfjs_poc/weights/reference.json", "utf-8"));
console.log(`Reference action_probs:`, ref.action_probs[0]);

// Load weights manually (no HTTP)
const manifestRaw = readFileSync("tfjs_poc/weights/model.json", "utf-8");
const manifest = JSON.parse(manifestRaw);
const shardBuffer = readFileSync("tfjs_poc/weights/group1-shard1of1.bin");

const weights = {};
let offset = 0;
for (const spec of manifest.weightsManifest[0].weights) {
  const numElements = spec.shape.reduce((a, b) => a * b, 1);
  const byteLength = numElements * 4;
  const data = new Float32Array(shardBuffer.buffer, shardBuffer.byteOffset + offset, numElements);
  weights[spec.name] = tf.tensor(Array.from(data), spec.shape, "float32");
  offset += byteLength;
}
weights._convStrides = cfg.conv_strides;
console.log(`Loaded ${Object.keys(weights).filter(k => !k.startsWith("_")).length} weight tensors`);

// Prepare input
const obs = tf.tensor(ref.obs, undefined, "int32");
const reward = tf.tensor(ref.reward, undefined, "float32");
const lastAction = tf.tensor(ref.last_action, undefined, "int32");
const time = tf.tensor(ref.time, undefined, "int32");

const timestep = { obs, reward, lastAction, time };

// Initialize KV caches
let kvCaches = initializeKvCaches(1, cfg);

// Run forward pass
console.log("Running forward pass...");
const t0 = performance.now();
const result = forward(timestep, kvCaches, weights, cfg);
const jsProbs = result.actionProbs.dataSync();
const elapsed = performance.now() - t0;
console.log(`Forward pass: ${elapsed.toFixed(1)}ms`);

// Compare
console.log("\nAction probabilities comparison:");
console.log("Action | JS           | JAX Ref      | Diff");
console.log("-------|--------------|--------------|--------");

const refProbs = ref.action_probs[0];
let maxDiff = 0;
for (let i = 0; i < refProbs.length; i++) {
  const diff = Math.abs(jsProbs[i] - refProbs[i]);
  maxDiff = Math.max(maxDiff, diff);
  const status = diff < 0.001 ? "OK" : diff < 0.01 ? "~" : "!!";
  console.log(`  ${i}    | ${jsProbs[i].toFixed(8)} | ${refProbs[i].toFixed(8)} | ${diff.toExponential(2)} ${status}`);
}

console.log(`\nMax absolute diff: ${maxDiff.toExponential(2)}`);
if (maxDiff < 0.01) {
  console.log("PASS");
} else {
  console.log("WARN: diff exceeds 0.01");
}

// Cleanup
result.actionProbs.dispose();
result.actionLogits.dispose();
for (const c of result.kvCaches) { c.key.dispose(); c.value.dispose(); }

process.exit(0);
