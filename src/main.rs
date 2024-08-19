use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use image::{ImageBuffer, Rgb};
use lazy_static::lazy_static;
use log::{error, info, warn};
use ndarray::{Array, Dim};
use num_bigint::{BigInt, ToBigInt};
use num_complex::Complex;
use num_traits::cast::ToPrimitive;
use petgraph::algo::dijkstra;
use petgraph::graph::{Graph, NodeIndex};
use rand::Rng;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio;

#[derive(Serialize, Deserialize, Clone)]
struct ComplexMessage<T> {
    content: T,
    timestamp: DateTime<Utc>,
    #[serde(with = "big_int_serialize")]
    complexity_factor: BigInt,
    meta_data: Vec<u8>,
}

mod big_int_serialize {
    use num_bigint::BigInt;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &BigInt, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.to_string().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigInt, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<BigInt>().map_err(serde::de::Error::custom)
    }
}

trait Printable: Send + Sync {
    fn to_string(&self) -> String;
    fn scramble(&mut self);
    fn unscramble(&mut self);
    fn apply_quantum_transformation(&mut self);
}

#[derive(Clone)]
enum MessageType {
    Greeting,
    Farewell,
    Nonsense,
    Quantum,
}

#[derive(Clone)]
struct MessagePrinter<T: Printable + Clone> {
    message: T,
    print_count: usize,
    graph: Graph<char, f32>,
}

impl<T: Printable + Clone> MessagePrinter<T> {
    fn new(message: T) -> Self {
        let mut graph = Graph::new();
        let nodes: Vec<NodeIndex> = message
            .to_string()
            .chars()
            .map(|c| graph.add_node(c))
            .collect();

        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], rand::random::<f32>());
            }
        }

        MessagePrinter {
            message,
            print_count: 0,
            graph,
        }
    }

    fn print(&mut self) {
        self.print_count += 1;
        let mut cloned_message = self.message.clone();
        cloned_message.scramble();
        cloned_message.unscramble();
        cloned_message.apply_quantum_transformation();

        let path = self.find_shortest_path();
        let message_str = path.into_iter().collect::<String>();

        println!("Print #{}: {}", self.print_count, message_str);
    }

    fn find_shortest_path(&self) -> Vec<char> {
        let start = self.graph.node_indices().next().unwrap();
        let end = self.graph.node_indices().last().unwrap();
        let distances = dijkstra(&self.graph, start, Some(end), |e| *e.weight());
        let mut path = vec![];
        let mut current = end;
        while current != start {
            path.push(self.graph[current]);
            if let Some(&next) = distances.get(&current) {
                let owo: NodeIndex = NodeIndex::new(next as usize);
                current = owo;
            } else {
                break;
            }
        }
        path.push(self.graph[start]);
        path.reverse();
        path
    }
}

impl Printable for ComplexMessage<String> {
    fn to_string(&self) -> String {
        format!(
            "{} (at {}) [Complexity: {}]",
            self.content, self.timestamp, self.complexity_factor
        )
    }

    fn scramble(&mut self) {
        self.content = self.content.chars().rev().collect::<String>();
    }

    fn unscramble(&mut self) {
        self.content = self.content.chars().rev().collect::<String>();
    }

    fn apply_quantum_transformation(&mut self) {
        self.content = self
            .content
            .chars()
            .map(|c| ((c as u8).wrapping_add(42) as char))
            .collect();
    }
}

macro_rules! create_message {
    ($content:expr) => {
        ComplexMessage {
            content: $content.to_string(),
            timestamp: Utc::now(),
            complexity_factor: rand::thread_rng()
                .gen_range(0..1000000)
                .to_bigint()
                .unwrap(),
            meta_data: vec![0; 1024],
        }
    };
}

macro_rules! perform_useless_calculation {
    ($x:expr) => {{
        let mut rng = rand::thread_rng();
        let random_float: f64 = rng.gen();
        let complex_num = Complex::new($x as f64, random_float);
        let result = (0..10000)
            .into_par_iter()
            .map(|i| {
                let fib = fibonacci(i);
                complex_num.powu(fib) * complex_num.sin()
            })
            .reduce(|| Complex::new(0.0, 0.0), |a, b| a + b);
        result.re.abs() as i32 % 4
    }};
}

macro_rules! choose_message_type {
    ($x:expr) => {
        match $x {
            0 => MessageType::Greeting,
            1 => MessageType::Farewell,
            2 => MessageType::Nonsense,
            _ => MessageType::Quantum,
        }
    };
}

lazy_static! {
    static ref HELLO_REGEX: Regex = Regex::new(r"^H.*o$").unwrap();
    static ref WORLD_REGEX: Regex = Regex::new(r"^W.*d!$").unwrap();
}

unsafe fn dangerous_string_manipulation(input: &str) -> String {
    let mut chars: Vec<u8> = input.bytes().collect();
    let matrix = Array::from_shape_fn(Dim([chars.len(), chars.len()]), |(i, j)| {
        (BigInt::from(i).pow(j as u32) % BigInt::from(256u32))
            .to_u8()
            .unwrap()
    });

    for (i, char) in chars.iter_mut().enumerate() {
        *char ^= matrix[[i, i]];
    }
    String::from_utf8_unchecked(chars)
}

fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn generate_useless_image(message: &str) {
    let (width, height) = (800, 600);
    let mut imgbuf = ImageBuffer::new(width, height);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let r = (x as f32 / width as f32 * 255.0) as u8;
        let g = (y as f32 / height as f32 * 255.0) as u8;
        let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
        *pixel = Rgb([r, g, b]);
    }

    let _ = imgbuf.save(format!("useless_image_{}.png", message.len()));
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let message_type = choose_message_type!(perform_useless_calculation!(42));

    let content = match message_type {
        MessageType::Greeting => "Hello, World!",
        MessageType::Farewell => "Goodbye, World!",
        MessageType::Nonsense => "Fnord, Xyzzy!",
        MessageType::Quantum => "Schrodinger's Cat!",
    };

    let mut complex_message = create_message!(content);
    let encrypted_message = unsafe { dangerous_string_manipulation(&complex_message.content) };
    let decrypted_message = unsafe { dangerous_string_manipulation(&encrypted_message) };

    complex_message.content = decrypted_message;

    if HELLO_REGEX.is_match(&complex_message.content)
        && WORLD_REGEX.is_match(&complex_message.content)
    {
        info!("Message appears to be a proper greeting");
    } else {
        warn!("Message does not conform to expected format");
    }

    let fib_sum: BigInt = (0..100).map(|n| BigInt::from(fibonacci(n))).sum();
    complex_message.complexity_factor += fib_sum;

    let printer = MessagePrinter::new(complex_message.clone());

    info!("Preparing to print message");

    let futures: Vec<_> = (0..10)
        .map(|_| {
            let mut printer_clone = printer.clone();
            tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    rand::random::<u64>() % 1000,
                ))
                .await;
                printer_clone.print();
            })
        })
        .collect();

    futures::future::join_all(futures).await;

    generate_useless_image(&complex_message.content);

    let meaningless_stream = stream::iter(0..1000000)
        .map(|i| async move { i * i })
        .buffer_unordered(100);

    let sum: u64 = meaningless_stream
        .fold(0, |acc, x| async move { acc + x })
        .await;

    error!("The meaningless sum is: {}", sum);
}
