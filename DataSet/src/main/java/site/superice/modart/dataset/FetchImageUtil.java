package site.superice.modart.dataset;

import com.google.gson.JsonParser;
import org.openqa.selenium.By;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.chrome.ChromeDriver;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Please run this class in the root directory of the project (ModArt).
 */
public final class FetchImageUtil {
    public static String TYPE = "tf-ingot";
    public static final Color BG = new Color(230, 230, 230);
    public static final Color TRANSPARENT = new Color(0, 0, 0, 0);

    @SuppressWarnings("BusyWait")
    public static void main(String[] args) {
        if (args.length > 0) {
            TYPE = args[0];
        }
        // Get gt5-rod.json from resource and parse it
        try (var is = FetchImageUtil.class.getResourceAsStream("/data/" + TYPE + ".json")) {
            Objects.requireNonNull(is);
            var json = JsonParser.parseReader(new InputStreamReader(is)).getAsJsonObject();
            var driver = new ChromeDriver();
            for (var entry : json.entrySet()) {
                var name = entry.getKey();
                var url = entry.getValue().getAsJsonObject().get("url").getAsString();
                System.out.println(url);
                System.out.println("Start fetching: " + name);
                driver.get(url);
                while (true) {
                    try {
                        var ele = driver.findElement(By.tagName("img"));
                        Thread.sleep(500);
                        var result = ele.getScreenshotAs(OutputType.BYTES);
                        // write to file
                        var path = Path.of("./DataSet/src/main/resources/color/", TYPE, name + ".png");
                        if (!Files.exists(path.getParent())) {
                            Files.createDirectories(path.getParent());
                        }
                        var grayPath = Path.of("./DataSet/src/main/resources/gray/", TYPE + "_gray", name + ".png");
                        if (!Files.exists(grayPath.getParent())) {
                            Files.createDirectories(grayPath.getParent());
                        }
                        var bufferedImage = ImageIO.read(new ByteArrayInputStream(result));
                        // resize to 16*16
                        var resizedImage = bufferedImage.getScaledInstance(16, 16, Image.SCALE_REPLICATE);
                        bufferedImage = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
                        bufferedImage.getGraphics().drawImage(resizedImage, 0, 0, TRANSPARENT, null);
                        // replace #E6E6E6 in "resized" with transparent
                        for (int i = 0; i < 16; i++) {
                            for (int j = 0; j < 16; j++) {
                                var rgb = bufferedImage.getRGB(i, j);
                                var r = (rgb >> 16) & 0xFF;
                                var g = (rgb >> 8) & 0xFF;
                                var b = rgb & 0xFF;
                                if (Math.abs(r - BG.getRed() + 1) < 3 && Math.abs(g - BG.getGreen() + 1) < 3 && Math.abs(b - BG.getBlue() + 1) < 3) {
                                    bufferedImage.setRGB(i, j, 0);
                                }
                            }
                        }
                        // write resizedImage
                        ImageIO.write(bufferedImage, "png", path.toFile());
                        // convert to grayscale image
                        var grayImage = new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB);
                        for (int i = 0; i < 16; i++) {
                            for (int j = 0; j < 16; j++) {
                                var color = new Color(bufferedImage.getRGB(i, j), true);
                                if (color.equals(TRANSPARENT)) {
                                    grayImage.setRGB(i, j, TRANSPARENT.getRGB());
                                } else {
                                    var gray = (int) (color.getRed() * 0.299 + color.getGreen() * 0.587 + color.getBlue() * 0.114);
                                    grayImage.setRGB(i, j, new Color(gray, gray, gray).getRGB());
                                }
                            }
                        }
                        // write grayImage
                        ImageIO.write(grayImage, "png", grayPath.toFile());
                        // success
                        System.out.println("Successfully fetched: " + name);
                        break;
                    } catch (Exception ignored) {
                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                    }
                }
            }
            driver.quit();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
