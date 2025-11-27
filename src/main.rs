use rand::Rng;

#[derive(Debug)]
struct NeuralNetwork<const SIZE: usize> {
    weights: [f64; SIZE],
    bias: f64,
    learning_rate: f64,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

impl<const SIZE: usize> NeuralNetwork<SIZE> {
    fn new() -> Self {
        let mut rng = rand::rng();
        let mut weights = [0.; SIZE];

        for weight in weights.iter_mut() {
            *weight = rng.random_range(0.0..1.0);
        }

        Self {
            weights,
            bias: rng.random_range(0.0..1.0),
            learning_rate: 0.1,
        }
    }

    fn predict(&self, input: &[f64; SIZE]) -> f64 {
        let mut sum = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            sum += input[i] * weight;
        }
        sigmoid(sum)
    }

    fn train(&mut self, inputs: Vec<[f64; SIZE]>, outputs: Vec<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (i, input) in inputs.iter().enumerate() {
                // Get a prediction for a given input
                let output = self.predict(input);

                // Compute the difference between the actual and the desired output
                let error = outputs[i] - output;

                // Find the gradient of the loss function
                // (sort of like a hint about the direction to adjust the weights in)
                let delta = derivative(output);

                // Adjust the weights and the bias to reduce error in the output
                for (weight, input) in self.weights.iter_mut().zip(input) {
                    *weight += self.learning_rate * error * *input * delta;
                }

                self.bias += self.learning_rate * error * delta;
            }
        }
    }
}

fn main() {
    println!("Hello, world!");
}
