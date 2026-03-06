// rust-grad: 一個微型自動微分引擎
// 靈感來自 Karpathy 的 micrograd，用 Rust 重新實現
// 支援 define-by-run 自動建圖 + 反向傳播

use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// 計算圖中的內部節點資料
struct ValueData {
    /// 前向值
    data: f64,
    /// 反向梯度 ∂L/∂self
    grad: f64,
    /// 產生這個節點的運算（用於 backward）
    backward_fn: Option<BackwardFn>,
    /// 上游節點（計算圖的邊）
    prev: Vec<Value>,
    /// 運算標籤（debug 用）
    op: String,
}

/// 反向傳播閉包
type BackwardFn = Rc<dyn Fn()>;

/// Value — 計算圖的節點
/// 每次運算自動記錄上游節點，backward() 時自動算梯度
#[derive(Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueData>>,
}

impl Value {
    /// 建立一個新的 Value 節點
    pub fn new(data: f64) -> Self {
        Value {
            inner: Rc::new(RefCell::new(ValueData {
                data,
                grad: 0.0,
                backward_fn: None,
                prev: vec![],
                op: String::new(),
            })),
        }
    }

    /// 讀取前向值
    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    /// 讀取梯度
    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    /// ReLU 激活函數
    /// forward: max(0, x)
    /// backward: x > 0 ? 1 : 0
    pub fn relu(&self) -> Value {
        let data = self.inner.borrow().data;
        let out_data = if data > 0.0 { data } else { 0.0 };

        let out = Value::new(out_data);
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = "relu".to_string();

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            let out_data = out_inner.borrow().data;
            self_inner.borrow_mut().grad += if out_data > 0.0 { out_grad } else { 0.0 };
        }));

        out
    }

    /// 次方運算
    /// forward: x^n
    /// backward: ∂(x^n)/∂x = n * x^(n-1)
    pub fn pow(&self, n: f64) -> Value {
        let data = self.inner.borrow().data;
        let out = Value::new(data.powf(n));
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = format!("**{}", n);

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            let self_data = self_inner.borrow().data;
            // ∂(x^n)/∂x = n * x^(n-1)
            self_inner.borrow_mut().grad += n * self_data.powf(n - 1.0) * out_grad;
        }));

        out
    }

    /// 指數函數
    /// forward: e^x
    /// backward: ∂(e^x)/∂x = e^x
    pub fn exp(&self) -> Value {
        let data = self.inner.borrow().data;
        let exp_val = data.exp();
        let out = Value::new(exp_val);
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = "exp".to_string();

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            // ∂(e^x)/∂x = e^x
            self_inner.borrow_mut().grad += exp_val * out_grad;
        }));

        out
    }

    /// 自然對數
    /// forward: ln(x)
    /// backward: ∂ln(x)/∂x = 1/x
    pub fn ln(&self) -> Value {
        let data = self.inner.borrow().data;
        let out = Value::new(data.ln());
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = "ln".to_string();

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            let self_data = self_inner.borrow().data;
            // ∂ln(x)/∂x = 1/x
            self_inner.borrow_mut().grad += (1.0 / self_data) * out_grad;
        }));

        out
    }

    /// tanh 激活函數
    /// forward: tanh(x)
    /// backward: ∂tanh(x)/∂x = 1 - tanh²(x)
    pub fn tanh(&self) -> Value {
        let data = self.inner.borrow().data;
        let t = data.tanh();
        let out = Value::new(t);
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = "tanh".to_string();

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            // ∂tanh(x)/∂x = 1 - tanh²(x)
            self_inner.borrow_mut().grad += (1.0 - t * t) * out_grad;
        }));

        out
    }

    /// sigmoid 激活函數
    /// forward: σ(x) = 1 / (1 + e^(-x))
    /// backward: ∂σ/∂x = σ(x)(1 - σ(x))
    pub fn sigmoid(&self) -> Value {
        let data = self.inner.borrow().data;
        let s = 1.0 / (1.0 + (-data).exp());
        let out = Value::new(s);
        out.inner.borrow_mut().prev = vec![self.clone()];
        out.inner.borrow_mut().op = "sigmoid".to_string();

        let self_inner = Rc::clone(&self.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            // ∂σ/∂x = σ(x)(1 - σ(x))
            self_inner.borrow_mut().grad += s * (1.0 - s) * out_grad;
        }));

        out
    }

    /// 反向傳播 — 從這個節點出發，計算所有上游節點的梯度
    /// 1. 拓撲排序（DFS）
    /// 2. 設定 self.grad = 1.0（∂L/∂L = 1）
    /// 3. 逆序走，逐個呼叫 backward_fn（鏈式法則）
    pub fn backward(&self) {
        // 拓撲排序
        let mut topo: Vec<Value> = vec![];
        let mut visited: Vec<*const RefCell<ValueData>> = vec![];

        fn build_topo(
            v: &Value,
            visited: &mut Vec<*const RefCell<ValueData>>,
            topo: &mut Vec<Value>,
        ) {
            let ptr = Rc::as_ptr(&v.inner);
            if visited.contains(&ptr) {
                return;
            }
            visited.push(ptr);
            for child in &v.inner.borrow().prev {
                build_topo(child, visited, topo);
            }
            topo.push(v.clone());
        }

        build_topo(self, &mut visited, &mut topo);

        // ∂L/∂L = 1
        self.inner.borrow_mut().grad = 1.0;

        // 逆序走拓撲排序，逐個算梯度
        for v in topo.iter().rev() {
            if let Some(ref f) = v.inner.borrow().backward_fn {
                f();
            }
        }
    }
}

// ===== Operator Overloading =====

/// Add: z = x + y
/// ∂z/∂x = 1, ∂z/∂y = 1
impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let out = Value::new(self.data() + other.data());
        out.inner.borrow_mut().prev = vec![self.clone(), other.clone()];
        out.inner.borrow_mut().op = "+".to_string();

        let self_inner = Rc::clone(&self.inner);
        let other_inner = Rc::clone(&other.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            self_inner.borrow_mut().grad += out_grad;
            other_inner.borrow_mut().grad += out_grad;
        }));

        out
    }
}

/// Mul: z = x * y
/// ∂z/∂x = y, ∂z/∂y = x
impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let self_data = self.data();
        let other_data = other.data();
        let out = Value::new(self_data * other_data);
        out.inner.borrow_mut().prev = vec![self.clone(), other.clone()];
        out.inner.borrow_mut().op = "*".to_string();

        let self_inner = Rc::clone(&self.inner);
        let other_inner = Rc::clone(&other.inner);
        let out_inner = Rc::clone(&out.inner);
        out.inner.borrow_mut().backward_fn = Some(Rc::new(move || {
            let out_grad = out_inner.borrow().grad;
            // ∂(x*y)/∂x = y
            self_inner.borrow_mut().grad += other_data * out_grad;
            // ∂(x*y)/∂y = x
            other_inner.borrow_mut().grad += self_data * out_grad;
        }));

        out
    }
}

/// Neg: -x
/// ∂(-x)/∂x = -1
impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Value {
        let minus_one = Value::new(-1.0);
        self * &minus_one
    }
}

/// Sub: z = x - y (實作為 x + (-y))
impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

/// Div: z = x / y (實作為 x * y^(-1))
/// ∂z/∂x = 1/y, ∂z/∂y = -x/y²
impl Div for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        self * &other.pow(-1.0)
    }
}

/// Display
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value(data={}, grad={})",
            self.data(),
            self.grad()
        )
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;

    // T02: Value::new 建立節點
    #[test]
    fn test_value_new() {
        let v = Value::new(3.0);
        assert_eq!(v.data(), 3.0);
        assert_eq!(v.grad(), 0.0);
    }

    // T04: Add forward
    #[test]
    fn test_add_forward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a + &b;
        assert_eq!(c.data(), 5.0);
    }

    // T05: Mul forward
    #[test]
    fn test_mul_forward() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = &a * &b;
        assert_eq!(c.data(), 12.0);
    }

    // T06: Neg forward
    #[test]
    fn test_neg_forward() {
        let a = Value::new(3.0);
        let b = -&a;
        assert_eq!(b.data(), -3.0);
    }

    // T07: Sub forward (a + (-b))
    #[test]
    fn test_sub_forward() {
        let a = Value::new(5.0);
        let b = Value::new(2.0);
        let c = &a + &(-&b);
        assert_eq!(c.data(), 3.0);
    }

    // T08: Pow forward
    #[test]
    fn test_pow_forward() {
        let a = Value::new(2.0);
        let b = a.pow(3.0);
        assert_eq!(b.data(), 8.0);
    }

    // T09: ReLU forward
    #[test]
    fn test_relu_forward_positive() {
        let a = Value::new(5.0);
        let b = a.relu();
        assert_eq!(b.data(), 5.0);
    }

    #[test]
    fn test_relu_forward_negative() {
        let a = Value::new(-3.0);
        let b = a.relu();
        assert_eq!(b.data(), 0.0);
    }

    // T10: backward add
    #[test]
    fn test_add_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a + &b;
        c.backward();
        assert_eq!(a.grad(), 1.0); // ∂c/∂a = 1
        assert_eq!(b.grad(), 1.0); // ∂c/∂b = 1
    }

    // T11: backward mul
    #[test]
    fn test_mul_backward() {
        let a = Value::new(3.0);
        let b = Value::new(4.0);
        let c = &a * &b;
        c.backward();
        assert_eq!(a.grad(), 4.0); // ∂c/∂a = b = 4
        assert_eq!(b.grad(), 3.0); // ∂c/∂b = a = 3
    }

    // T12: backward pow
    #[test]
    fn test_pow_backward() {
        let a = Value::new(2.0);
        let b = a.pow(3.0);
        b.backward();
        assert_eq!(a.grad(), 12.0); // ∂(x^3)/∂x = 3x^2 = 3*4 = 12
    }

    // T13: backward relu
    #[test]
    fn test_relu_backward_positive() {
        let a = Value::new(5.0);
        let b = a.relu();
        b.backward();
        assert_eq!(a.grad(), 1.0);
    }

    #[test]
    fn test_relu_backward_negative() {
        let a = Value::new(-3.0);
        let b = a.relu();
        b.backward();
        assert_eq!(a.grad(), 0.0);
    }

    // T14: 鏈式法則 — 多層組合
    #[test]
    fn test_chain_rule() {
        // f = (a * b) + c
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(4.0);
        let d = &a * &b; // d = 6
        let e = &d + &c; // e = 10
        e.backward();
        assert_eq!(a.grad(), 3.0); // ∂e/∂a = b = 3
        assert_eq!(b.grad(), 2.0); // ∂e/∂b = a = 2
        assert_eq!(c.grad(), 1.0); // ∂e/∂c = 1
    }

    // T15: 共用節點 — 梯度累加
    #[test]
    fn test_shared_node_grad_accumulation() {
        let a = Value::new(3.0);
        let b = &a + &a; // b = 2a = 6
        b.backward();
        assert_eq!(b.data(), 6.0);
        assert_eq!(a.grad(), 2.0); // ∂(2a)/∂a = 2
    }

    // T17: exp forward + backward
    #[test]
    fn test_exp_forward() {
        let a = Value::new(1.0);
        let b = a.exp();
        assert!((b.data() - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_exp_backward() {
        let a = Value::new(2.0);
        let b = a.exp();
        b.backward();
        assert!((a.grad() - 2.0_f64.exp()).abs() < 1e-6);
    }

    // T17: ln forward + backward
    #[test]
    fn test_ln_forward() {
        let a = Value::new(std::f64::consts::E);
        let b = a.ln();
        assert!((b.data() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ln_backward() {
        let a = Value::new(3.0);
        let b = a.ln();
        b.backward();
        assert!((a.grad() - 1.0 / 3.0).abs() < 1e-6);
    }

    // T18: div forward + backward
    #[test]
    fn test_div_forward() {
        let a = Value::new(6.0);
        let b = Value::new(3.0);
        let c = &a / &b;
        assert!((c.data() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_div_backward() {
        let a = Value::new(6.0);
        let b = Value::new(3.0);
        let c = &a / &b;
        c.backward();
        assert!((a.grad() - 1.0 / 3.0).abs() < 1e-6);
        assert!((b.grad() - (-2.0 / 3.0)).abs() < 1e-6);
    }

    // T18: sub operator
    #[test]
    fn test_sub_op_forward() {
        let a = Value::new(5.0);
        let b = Value::new(2.0);
        let c = &a - &b;
        assert_eq!(c.data(), 3.0);
    }

    #[test]
    fn test_sub_backward() {
        let a = Value::new(5.0);
        let b = Value::new(2.0);
        let c = &a - &b;
        c.backward();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
    }

    // T21: tanh
    #[test]
    fn test_tanh_forward() {
        let a = Value::new(0.0);
        assert!((a.tanh().data() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_backward() {
        let a = Value::new(1.0);
        let b = a.tanh();
        b.backward();
        let t = 1.0_f64.tanh();
        assert!((a.grad() - (1.0 - t * t)).abs() < 1e-6);
    }

    // T21: sigmoid
    #[test]
    fn test_sigmoid_forward() {
        let a = Value::new(0.0);
        assert!((a.sigmoid().data() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_backward() {
        let a = Value::new(1.0);
        let b = a.sigmoid();
        b.backward();
        let s = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((a.grad() - s * (1.0 - s)).abs() < 1e-6);
    }

    // T16: micrograd 官方範例 — 完整版（含 div）
    #[test]
    fn test_micrograd_example() {
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

        // micrograd 的答案：g ≈ 24.7041, ∂g/∂a ≈ 138.8338, ∂g/∂b ≈ 645.5773
        assert!((g.data() - 24.70408163265306).abs() < 1e-4);
        g.backward();
        assert!((a.grad() - 138.83381924198252).abs() < 1e-4);
        assert!((b.grad() - 645.5772594752186).abs() < 1e-4);
    }
}
