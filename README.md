# rust-grad 🦀

一個微型自動微分引擎，用 Rust 從零實現。

> Define-by-run 自動建圖 + 反向傳播，~200 行 Rust。

---

## 快速開始

```rust
use rust_grad::Value;

let a = Value::new(-4.0);
let b = Value::new(2.0);

// 前向：任意組合運算，自動建立計算圖
let c = &a + &b;
let d = &(&a * &b) + &b.pow(3.0);
let e = &c - &d;
let f = e.pow(2.0);

// 反向：一行算出所有梯度
f.backward();

println!("∂f/∂a = {}", a.grad());
println!("∂f/∂b = {}", b.grad());
```

---

## 支援的運算

| 運算 | 用法 | forward | backward (∂/∂x) |
|------|------|---------|------------------|
| 加法 | `&a + &b` | x + y | 1 |
| 減法 | `&a - &b` | x - y | 1, -1 |
| 乘法 | `&a * &b` | x × y | y, x |
| 除法 | `&a / &b` | x / y | 1/y, -x/y² |
| 負號 | `-&a` | -x | -1 |
| 次方 | `a.pow(n)` | xⁿ | n·xⁿ⁻¹ |
| 指數 | `a.exp()` | eˣ | eˣ |
| 對數 | `a.ln()` | ln(x) | 1/x |
| ReLU | `a.relu()` | max(0,x) | x>0 ? 1 : 0 |
| tanh | `a.tanh()` | tanh(x) | 1 - tanh²(x) |
| sigmoid | `a.sigmoid()` | σ(x) | σ(x)(1-σ(x)) |

---

## 核心概念

### 1. Value — 計算圖的節點

每個 `Value` 儲存：
- **data** — 前向值
- **grad** — 反向梯度（∂Loss/∂self）
- **prev** — 上游節點（計算圖的邊）
- **backward_fn** — 這個 op 的局部梯度計算

### 2. 自動建圖

每次運算（`+`, `*`, `.relu()` 等）會自動：
1. 計算輸出值（forward）
2. 記錄輸入節點到 `prev`（建邊）
3. 綁定 `backward_fn`（記錄偏微分規則）

```
a ──┐
    ├── [*] ──> c ──┐
b ──┘               ├── [+] ──> e
            d ──────┘
```

### 3. 反向傳播

呼叫 `e.backward()` 時：
1. 從 `e` 出發，DFS 拓撲排序
2. 設定 `e.grad = 1.0`（∂e/∂e = 1）
3. 逆序走，每個節點呼叫自己的 `backward_fn`
4. 梯度透過鏈式法則自動累加回所有葉節點

```
e.grad = 1.0
    ↓ backward_fn（鏈式法則）
c.grad += ..., d.grad += ...
    ↓
a.grad += ..., b.grad += ...
```

---

## 測試

```bash
cargo test
```

28 個測試覆蓋所有 op 的 forward + backward，包含 micrograd 官方範例的數值驗證。

---

## 設計哲學

- **最小化**：只做自動微分，不做 tensor、不做 nn layer
- **可讀性**：每個 op 的梯度公式直接寫在程式碼裡
- **正確性**：TCR 開發模式（Test && Commit || Revert），repo 永遠是綠的
- **可擴展**：要組 MLP、CNN、Transformer？import rust-grad 自己蓋

---

## License

MIT
