package texture.util;

import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;

public final class ImageUtils {
    private ImageUtils() {
        throw new UnsupportedOperationException();
    }

    public static @NotNull BufferedImage rotate(@NotNull BufferedImage image, int degree) {
        var w = image.getWidth();
        var h = image.getHeight();
        var rotated = new BufferedImage(w, h, image.getType());
        var g = rotated.createGraphics();
        g.rotate(Math.toRadians(degree), w / 2.0, h / 2.0);
        g.drawImage(image, null, 0, 0);
        return rotated;
    }

    public static @NotNull BufferedImage mirrorLR(@NotNull BufferedImage image) {
        var w = image.getWidth();
        var h = image.getHeight();
        var mirrored = new BufferedImage(w, h, image.getType());
        var g = mirrored.createGraphics();
        g.drawImage(image, 0, 0, w, h, w, 0, 0, h, null);
        return mirrored;
    }

    public static @NotNull BufferedImage mirrorUD(@NotNull BufferedImage image) {
        var w = image.getWidth();
        var h = image.getHeight();
        var mirrored = new BufferedImage(w, h, image.getType());
        var g = mirrored.createGraphics();
        g.drawImage(image, 0, 0, w, h, 0, h, w, 0, null);
        return mirrored;
    }
}
