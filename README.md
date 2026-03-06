# rust-grad 🦀

一個微型自動微分引擎，用 Rust 從零實現。

> Define-by-run 自動建圖 + 反向傳播，~200 行 Rust。

---

## 安裝與使用

### 前置需求

```bash
# 安裝 Rust（如果還沒裝）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 確認安裝成功
rustc --version
cargo --version
```

### 方法一：Clone 下來跑測試

```bash
git clone https://github.com/ryansoq/rust-grad.git
cd rust-grad

# 跑所有測試（28 個，涵蓋每個 op 的 forward + backward）
cargo test

# 跑範例
cargo run --example basic
```

### 方法二：作為依賴引入你的專案

```bash
# 建立新專案
cargo new my-ml-project
cd my-ml-project
```

在 `Cargo.toml` 加入：

```toml
[dependencies]
rust-grad = { git = "https://github.com/ryansoq/rust-grad.git" }
```

在 `src/main.rs` 使用：

```rust
use rust_grad::Value;

fn main() {
    let a = Value::new(-4.0);
    let b = Value::new(2.0);

    // 前向：任意組合運算，自動建立計算圖
    let c = &a + &b;                       // c = -2
    let d = &(&a * &b) + &b.pow(3.0);      // d = -8 + 8 = 0
    let e = &c - &d;                        // e = -2
    let f = e.pow(2.0);                     // f = 4

    // 反向：一行算出所有梯度
    f.backward();

    println!("f = {}", f.data());           // 4.0
    println!("∂f/∂a = {}", a.grad());       // df/da
    println!("∂f/∂b = {}", b.grad());       // df/db
}
```

```bash
cargo run
```

### Rust 新手提示

- `&a + &b` — Rust 用 `&`（借用）避免移動所有權，這樣 `a`, `b` 之後還能用
- `.backward()` — 反向傳播，算完後用 `.grad()` 讀梯度
- `cargo test` — Rust 內建測試框架，測試寫在 `#[cfg(test)]` 模組裡

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

## 開發流程：AI-TDD + TCR

這個專案採用 **TCR（Test && Commit || Revert）** 開發模式：

```bash
cargo test && git commit -m "TCR: working" || git reset --hard
```

**測試通過 → 自動 commit。測試失敗 → 直接砍掉重來。**

不修 bug、不堆垃圾邏輯。每個 commit 都是綠的。

### 開發順序

每個功能都是「先寫測試，再寫實作」：

```
Step 1: 寫測試 → test_add_forward（預期 2+3=5）
Step 2: 寫最少量的實作讓測試通過
Step 3: cargo test && commit || revert
Step 4: 下一個測試...
```

### 實際 commit 歷程

```
T01:     cargo init
T02-T16: Value + add/mul/neg/pow/relu + backward + chain rule + micrograd 驗證
T17-T21: exp/ln/div/sub/tanh/sigmoid + micrograd 完整範例通過
T20:     README
```

共 28 個測試，零次 revert。

### TCR 腳本

專案內附 `tcr.sh`：

```bash
./tcr.sh "commit message"
```

---

## 測試

```bash
# 跑所有測試
cargo test

# 跑特定測試
cargo test test_micrograd_example

# 看測試輸出
cargo test -- --nocapture
```

28 個測試覆蓋：
- 每個 op 的 forward 正確性
- 每個 op 的 backward 梯度正確性
- 鏈式法則（多層組合）
- 共用節點的梯度累加
- micrograd 官方範例的數值驗證（精確到小數後 4 位）

---

## 設計哲學

- **最小化**：只做自動微分，不做 tensor、不做 nn layer
- **可讀性**：每個 op 的梯度公式直接寫在程式碼註解裡
- **正確性**：TCR 開發，repo 永遠是綠的
- **可擴展**：要組 MLP、CNN、Transformer？import rust-grad 自己蓋

---

## License

MIT
