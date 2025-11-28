mod data;

use rand::Rng;

use crate::data::TestData;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn dsigmoid(y: f64) -> f64 {
    y * (1.0 - y)
}

struct NeuralNetwork<const INPUT: usize, const HIDDEN: usize> {
    w1: [[f64; INPUT]; HIDDEN], // Input → Hidden
    b1: [f64; HIDDEN],

    w2: [f64; HIDDEN], // Hidden → Output
    b2: f64,

    lr: f64,
}

impl<const INPUT: usize, const HIDDEN: usize> NeuralNetwork<INPUT, HIDDEN> {
    fn new() -> Self {
        let mut rng = rand::rng();

        // 初始化 w1
        let mut w1 = [[0.0; INPUT]; HIDDEN];
        for weights in w1.iter_mut() {
            for w in weights.iter_mut() {
                *w = rng.random_range(-1.0..1.0)
            }
        }

        // 初始化 w2
        let mut w2 = [0.0; HIDDEN];
        for w in w2.iter_mut() {
            *w = rng.random_range(-1.0..1.0);
        }

        Self {
            w1,
            b1: [0.0; HIDDEN],
            w2,
            b2: 0.0,
            lr: 0.1,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn predict(&self, x: &[f64; INPUT]) -> f64 {
        // Hidden layer
        let mut h = [0.0; HIDDEN];
        for j in 0..HIDDEN {
            let mut sum = self.b1[j];
            for i in 0..INPUT {
                sum += self.w1[j][i] * x[i];
            }
            h[j] = sigmoid(sum);
        }

        // Output layer (linear)
        let mut y = self.b2;
        for j in 0..HIDDEN {
            y += self.w2[j] * h[j];
        }
        y
    }

    #[allow(clippy::needless_range_loop)]
    fn train(&mut self, inputs: &[[f64; INPUT]], outputs: &[f64], epochs: usize) {
        for _ in 0..epochs {
            for (idx, x) in inputs.iter().enumerate() {
                let target = outputs[idx];

                // ---- Forward pass ----
                let mut h = [0.0; HIDDEN];
                let mut h_sum = [0.0; HIDDEN];

                for j in 0..HIDDEN {
                    h_sum[j] = self.b1[j];
                    for i in 0..INPUT {
                        h_sum[j] += self.w1[j][i] * x[i];
                    }
                    h[j] = sigmoid(h_sum[j]);
                }

                let mut y = self.b2;
                for j in 0..HIDDEN {
                    y += self.w2[j] * h[j];
                }

                let error = target - y;

                // ---- Backprop output layer ----
                for j in 0..HIDDEN {
                    self.w2[j] += self.lr * error * h[j];
                }
                self.b2 += self.lr * error;

                // ---- Backprop hidden layer ----
                for j in 0..HIDDEN {
                    let grad = error * self.w2[j] * dsigmoid(h[j]);
                    for i in 0..INPUT {
                        self.w1[j][i] += self.lr * grad * x[i];
                    }
                    self.b1[j] += self.lr * grad;
                }
            }
        }
    }
}

fn main() {
    let (tran_assert, test_inputs) = data::get_data("data.csv").unwrap();
    let inputs = tran_assert.inputs;
    let outputs = tran_assert.outputs;
    let mut model: NeuralNetwork<2, 4> = NeuralNetwork::new();

    model.train(&inputs, &outputs, 100000);

    for TestData { input, output } in test_inputs.iter() {
        // Pass a set of inputs (two numbers) and get a prediction back which should be a sum of the two numbers
        let prediction = model.predict(input);
        println!(
            "Input: {:?}, Prediction: {:.2}, result: {output}",
            input, prediction
        );
    }
}
