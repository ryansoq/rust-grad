use rust_grad::Value;
use std::fs;
use std::process::Command;

fn main() {
    // 簡單的神經元：y = tanh(w1*x1 + w2*x2 + b)
    let x1 = Value::new(2.0).label("x1");
    let x2 = Value::new(0.0).label("x2");
    let w1 = Value::new(-3.0).label("w1");
    let w2 = Value::new(1.0).label("w2");
    let b  = Value::new(6.88).label("b");

    let x1w1 = (&x1 * &w1).label("x1*w1");
    let x2w2 = (&x2 * &w2).label("x2*w2");
    let sum  = (&x1w1 + &x2w2).label("sum");
    let n    = (&sum + &b).label("n");
    let o    = n.tanh().label("o");

    o.backward();

    println!("{}", o);

    // 產生 DOT
    let dot = o.to_dot();
    fs::write("neuron.dot", &dot).expect("write dot");

    // 嘗試產生 PNG
    match Command::new("dot").args(["-Tpng", "neuron.dot", "-o", "neuron.png"]).status() {
        Ok(s) if s.success() => println!("📊 Graph saved: neuron.png"),
        _ => {
            println!("📊 DOT saved: neuron.dot");
            println!("   Run: dot -Tpng neuron.dot -o neuron.png");
        }
    }

    // 嘗試產生 SVG
    if let Ok(s) = Command::new("dot").args(["-Tsvg", "neuron.dot", "-o", "neuron.svg"]).status() {
        if s.success() {
            println!("📊 SVG saved: neuron.svg");
        }
    }
}
