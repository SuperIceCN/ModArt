package texture.util;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public final class LazyData<T> {
    private final AtomicReference<T> data = new AtomicReference<>();
    private final Supplier<T> supplier;

    public LazyData(Supplier<T> supplier) {
        this.supplier = supplier;
    }

    public T get() {
        return data.updateAndGet(v -> v == null ? supplier.get() : v);
    }

    public T update() {
        return data.updateAndGet(v -> supplier.get());
    }
}
