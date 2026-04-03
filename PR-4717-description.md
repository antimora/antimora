## Pull Request Template

### Checklist

- [ ] Confirmed that `cargo run-checks` command has been executed.
- [ ] Made sure the book is up to date with changes in this PR.

### Related Issues/PRs

- Closes #3628 — Default backend / remove boilerplate `<B: Backend>` generic (core motivation, by @nathanielsimard)
- Related #4415 — Runtime backend selection for cross-platform/Python interop
- Related #705 — Backend is not object-safe; runtime dispatch requires structural change
- Related #2276 — Multi-backend decorator via device enum
- Preparatory: #4508 (introduces `burn-dispatch` / `Dispatch` backend), #4629 (autodiff checkpointing as device property), #4653 (dtype from device settings), #4666 (tests migrated to `Dispatch`)

### Changes

**Problem:** Every piece of user code that uses tensors must carry a `B: Backend` type parameter, which propagates through every struct, function, and trait impl in a project:

```rust
// Before: B infects the entire call stack
pub struct Model<B: Backend> { layer: nn::Linear<B>, ... }
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> { ... }
}
fn run<B: AutodiffBackend>(device: B::Device) { ... }
```

This creates three concrete problems:

1. **Boilerplate:** Every library/app must expose `<B: Backend>` generics or lock users into one backend.
2. **No runtime dispatch:** Backend is compile-time only; can't fall back from GPU→CPU when hardware is unavailable, and switching devices requires going through `TensorData` manipulations (which is really meant to be a data representation, not a device-transfer mechanism).
3. **Autodiff coupling:** Training requires a separate `Autodiff<B>` wrapper type, making the backend type even more complex.

**Solution:** Two changes working together:

- **`burn-dispatch` crate** (landed in #4508) provides a single concrete `Dispatch` backend that implements the `Backend` trait via compile-time enum dispatch over all enabled backends. Backends are still behind feature flags, so users enable only what they need. `DispatchDevice` and `DispatchTensor` are enums over per-backend device/tensor types, so the actual backend is selected at runtime from the enum variant while the type system sees only `Dispatch`.

- **Remove `B` from `Tensor`:** Since `Dispatch` is the one backend for user-facing code, `Tensor<B, D, K>` becomes `Tensor<D, K>`. Autodiff is now a property of the `Device` rather than a type parameter — call `.autodiff()` on any device to opt into gradient tracking.

```rust
// After: no backend generic anywhere in user code
let device = Device::default();            // auto-selects best available backend
let device = Device::default().autodiff(); // enables gradient tracking

let x = Tensor::<2>::zeros([3, 4], &device);
```

The `DispatchDevice` enum dispatches based on which Cargo feature flags are enabled (`cuda`, `ndarray`, `vulkan`, etc.). When only one backend feature is enabled the compiler optimizes the `match` away entirely; with multiple backends enabled the overhead is minimal enum dispatch rather than vtable dispatch.

**Key benefits this unlocks:**

- Easy runtime switching between backend devices (e.g. CPU ↔ GPU) without `TensorData` round-trips.
- Simpler development cycles — feature-gate the primitive to keep compile times fast while iterating.
- A path toward making the primitive opaque, further improving compile times.
- Docs and book will be updated separately to guide existing users through the migration.

### Testing

Backend tests were migrated to use `Dispatch` in #4666 and validate correctness across all backends.
