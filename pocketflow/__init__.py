class _ActionRoute:
    def __init__(self, node, action):
        self.node = node
        self.action = action
    def __rshift__(self, target):
        self.node.routes[self.action] = target
        return target

class Node:
    """Minimal PocketFlow Node contract used for tests."""
    def __init__(self, max_retries: int = 1, wait: int = 0):
        self.max_retries = max_retries
        self.wait = wait
        self.next = None
        self.routes = {}
        self.params = {}
    def prep(self, shared):
        return None
    def exec(self, prep_res):
        return None
    def post(self, shared, prep_res, exec_res):
        return None
    def run(self, shared):
        prep_res = self.prep(shared)
        exec_res = self.exec(prep_res)
        return self.post(shared, prep_res, exec_res)
    def __rshift__(self, other):
        self.next = other
        return other
    def __sub__(self, action: str):
        return _ActionRoute(self, action)

class Flow(Node):
    """Simplified flow that executes connected nodes sequentially."""
    def __init__(self, start: Node):
        super().__init__()
        self.start = start
    def run(self, shared):
        node = self.start
        while node is not None:
            action = node.run(shared)
            if action in getattr(node, 'routes', {}):
                node = node.routes[action]
            else:
                node = getattr(node, 'next', None)
        return None

class BatchFlow(Flow):
    """Batch version executing the start node with different params."""
    def run(self, shared):
        batch_params = self.prep(shared)
        for params in batch_params:
            self.start.params = params
            super().run(shared)
        return None
