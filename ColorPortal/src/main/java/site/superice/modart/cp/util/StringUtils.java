package site.superice.modart.cp.util;

import org.jetbrains.annotations.NotNull;

public final class StringUtils {
    private StringUtils() {
        throw new AssertionError("No instance for you!");
    }

    @NotNull
    public static String beforeLast(@NotNull String str, String delimiter) {
        int index = str.lastIndexOf(delimiter);
        if (index == -1) {
            return str;
        }
        return str.substring(0, index);
    }

    @NotNull
    public static String afterFirst(@NotNull String str, String delimiter) {
        int index = str.indexOf(delimiter);
        if (index == -1) {
            return str;
        }
        return str.substring(index + 1);
    }
}
