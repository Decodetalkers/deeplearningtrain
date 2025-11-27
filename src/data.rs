use std::path::Path;

use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;

use rand::Rng;

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
