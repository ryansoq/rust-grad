use rust_grad::Value;

fn main() {
    println!("=== rust-grad 基本範例 ===\n");

    // 1. 簡單運算
    println!("--- 1. 基本四則運算 ---");
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = &a + &b;        // 5
    let d = &a * &b;        // 6
    let e = &c - &d;        // -1
    println!("a={}, b={}", a.data(), b.data());
    println!("a+b = {}", c.data());
    println!("a*b = {}", d.data());
    println!("(a+b)-(a*b) = {}", e.data());

    // 2. 反向傳播
    println!("\n--- 2. 反向傳播（梯度計算）---");
    let x = Value::new(3.0);
    let y = Value::new(4.0);
    let z = &x * &y;        // z = 12
    z.backward();
    println!("z = x * y = {} * {} = {}", x.data(), y.data(), z.data());
    println!("∂z/∂x = {} (應該 = y = 4)", x.grad());
    println!("∂z/∂y = {} (應該 = x = 3)", y.grad());

    // 3. 鏈式法則
    println!("\n--- 3. 鏈式法則 ---");
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = &a * &b;        // c = 6
    let d = c.pow(2.0);     // d = 36
    d.backward();
    // ∂d/∂a = ∂(a*b)²/∂a = 2(a*b)*b = 2*6*3 = 36
    println!("d = (a*b)² = ({} * {})² = {}", a.data(), b.data(), d.data());
    println!("∂d/∂a = {} (應該 = 2*a*b² = 36)", a.grad());
    println!("∂d/∂b = {} (應該 = 2*a²*b = 24)", b.grad());

    // 4. 激活函數
    println!("\n--- 4. 激活函數 ---");
    let x = Value::new(-2.0);
    println!("x = {}", x.data());
    println!("relu(x) = {}", x.relu().data());
    println!("sigmoid(x) = {:.4}", x.sigmoid().data());
    println!("tanh(x) = {:.4}", x.tanh().data());

    let x = Value::new(2.0);
    println!("\nx = {}", x.data());
    println!("relu(x) = {}", x.relu().data());
    println!("sigmoid(x) = {:.4}", x.sigmoid().data());
    println!("tanh(x) = {:.4}", x.tanh().data());

    // 5. micrograd 官方範例
    println!("\n--- 5. micrograd 官方範例驗證 ---");
    let a = Value::new(-4.0);
    let b = Value::new(2.0);
    let c = &a + &b;
    let d = &(&a * &b) + &b.pow(3.0);
    let c = &c + &(&c + &Value::new(1.0));
    let c = &c + &(&(&Value::new(1.0) + &c) + &(-&a));
    let d2 = &d * &Value::new(2.0);
    let ba_relu = (&b + &a).relu();
    let d = &d + &(&d2 + &ba_relu);
    let d3 = &Value::new(3.0) * &d;
    let bna_relu = (&b + &(-&a)).relu();
    let d = &d + &(&d3 + &bna_relu);
    let e = &c - &d;
    let f = e.pow(2.0);
    let g = &f / &Value::new(2.0);
    let g = &g + &(&Value::new(10.0) / &f);

    println!("g = {:.4} (micrograd: 24.7041)", g.data());
    g.backward();
    println!("∂g/∂a = {:.4} (micrograd: 138.8338)", a.grad());
    println!("∂g/∂b = {:.4} (micrograd: 645.5773)", b.grad());

    println!("\n✅ 完成！");
}
