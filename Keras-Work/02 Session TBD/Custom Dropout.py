class MyDropout(keras.layers.Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        # Use seed_generator for managing RNG state.
        # It is a state element and its seed variable is
        # tracked as part of `layer.variables`.
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        # Use `keras.random` for random ops.
        return keras.random.dropout(inputs, self.rate, seed=self.seed_generator)