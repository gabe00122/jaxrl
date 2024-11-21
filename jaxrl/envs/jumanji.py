class State:
    def __init__(self):
        pass


class Transition:
    def __init__(self):
        pass


class Action:
    def __init__(self):
        pass


class JumanjiWrapper:
    def __init__(self):
        self.is_jittable = False
    
    def reset(self) -> tuple[State, Transition]:
        return State(), Transition()

    def step(self, state: State, action: Transition) -> tuple[State, Transition]:
        return state, Transition()
