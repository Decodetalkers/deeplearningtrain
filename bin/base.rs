use rand::Rng;

mod data {
    use std::path::Path;

    use rand::Rng;
    use serde::Deserialize;
    use std::fs::File;
    use std::io::BufReader;

    #[derive(Debug, Deserialize)]
    struct ReadData {
        left: f64,
        right: f64,
        result: f64,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error("IoError")]
        IoError(#[from] std::io::Error),
        #[error("CSV Error")]
        CsvSerde(#[from] csv::Error),
    }

    fn read_csv<P: AsRef<Path>>(path: P) -> Result<Vec<ReadData>, Error> {
        let mut data = vec![];
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut rdr = csv::Reader::from_reader(reader);

        for re in rdr.deserialize() {
            let record: ReadData = re?;
            data.push(record);
        }
        Ok(data)
    }

    pub struct TranData {
        pub inputs: Vec<[f64; 2]>,
        pub outputs: Vec<f64>,
    }

    impl TranData {
        fn new() -> Self {
            Self {
                inputs: vec![],
                outputs: vec![],
            }
        }
    }

    pub struct TestData {
        pub input: [f64; 2],
        pub output: f64,
    }

    pub fn get_data<P: AsRef<Path>>(path: P) -> Result<(TranData, Vec<TestData>), Error> {
        let data_sets = read_csv(path)?;
        let mut rng = rand::rng();

        let mut tran_asserts = TranData::new();
        let mut test_asserts = vec![];

        for ReadData {
            left,
            right,
            result,
        } in data_sets
        {
            let dice = rng.random_range(0.0..1.0);
            if dice < 0.66 {
                tran_asserts.inputs.push([left, right]);
                tran_asserts.outputs.push(result);
            } else {
                test_asserts.push(TestData {
                    input: [left, right],
                    output: result,
                });
            }
        }

        Ok((tran_asserts, test_asserts))
    }
}
use data::TestData;

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

    fn predict(&self, x: &[f64; INPUT]) -> f64 {
        let (_, y) = self.forward(x);
        y
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, x: &[f64; INPUT]) -> ([f64; HIDDEN], f64) {
        // ---- Forward pass ----
        let mut h = [0.0; HIDDEN];

        for j in 0..HIDDEN {
            let mut sum = self.b1[j];
            for i in 0..INPUT {
                sum += self.w1[j][i] * x[i];
            }
            h[j] = sigmoid(sum);
        }

        let mut y = self.b2;
        for j in 0..HIDDEN {
            y += self.w2[j] * h[j];
        }
        (h, y)
    }

    #[allow(clippy::needless_range_loop)]
    fn train(&mut self, inputs: &[[f64; INPUT]], outputs: &[f64], epochs: usize) {
        for _ in 0..epochs {
            for (idx, x) in inputs.iter().enumerate() {
                let target = outputs[idx];

                // ---- Forward pass ----
                let (h, y) = self.forward(x);

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
