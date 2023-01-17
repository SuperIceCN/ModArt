package site.superice.modart.cp.data;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.gson.JsonParser;
import de.androidpit.colorthief.ColorThief;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import site.superice.modart.dataset.DataSetMain;
import site.superice.modart.cp.util.ImageUtils;
import site.superice.modart.cp.util.StringUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class EchoDataSet extends RandomAccessDataset {
    public static Map<String, BufferedImage> IMAGE_CACHE = new ConcurrentHashMap<>();

    public List<String> init() {
        // row: type, column: material, value: image_path
        try (var imageTypeListInputStream = DataSetMain.class.getResourceAsStream("/data/data.json")) {
            Objects.requireNonNull(imageTypeListInputStream);
            var imageTypeArr = JsonParser.parseReader(new InputStreamReader(imageTypeListInputStream)).getAsJsonArray();
            for (var each : imageTypeArr) {
                var type = StringUtils.afterFirst(each.getAsString(), "-");
                try (var imageListInputStream = DataSetMain.class.getResourceAsStream("/data/" + each.getAsString() + ".json")) {
                    Objects.requireNonNull(imageListInputStream);
                    var imageObj = JsonParser.parseReader(new InputStreamReader(imageListInputStream)).getAsJsonObject();
                    for (var image : imageObj.entrySet()) {
                        var material = StringUtils.beforeLast(image.getKey()
                                .replace("Small ", "")
                                .replace("Tiny ", "")
                                .replace("Double ", "")
                                .replace("Long ", ""), " ");
                        imageTable.put(type, material, "/color/" + each.getAsString() + "/" + image.getKey() + ".png");
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        var rows = imageTable.rowKeySet().stream().toList();
        var columns = imageTable.columnKeySet().stream().toList();
        var allImageEntries = new ArrayList<String>(rows.size() * columns.size());
        for (var row : rows) {
            for (var column : columns) {
                var imagePathStr = imageTable.get(row, column);
                if (imagePathStr != null) {
                    allImageEntries.add(imagePathStr);
                }
            }
        }
        return allImageEntries;
    }

    private final Table<String, String, String> imageTable = HashBasedTable.create();
    private final List<String> imagePathEntries;
    private final List<ImagePair> imageDataEntries;
    private final int length;

    public EchoDataSet(BaseBuilder<?> builder, int length) {
        super(builder);
        imagePathEntries = init();
        imageDataEntries = new CopyOnWriteArrayList<>();
        this.length = length;
    }

    @Override
    public Record get(NDManager manager, long index) {
        var imageData = imageDataEntries.get((int) index);
        var dataTensor = NDImageUtils.toTensor(toNDArray(manager, imageData.image(), Image.Flag.COLOR));
        dataTensor.setRequiresGradient(true);
        var dataNDList = new NDList(dataTensor);
        var labelNDList = new NDList(NDImageUtils.toTensor(toNDArray(manager, imageData.label(), Image.Flag.COLOR)));
        return new Record(dataNDList, labelNDList);
    }

    /**
     * @param flag null -> 4 channels
     */
    public static NDArray toNDArray(@NotNull NDManager manager, @NotNull BufferedImage image, @Nullable Image.Flag flag) {
        int width = image.getWidth();
        int height = image.getHeight();
        int channel = flag == null ? 4 : switch (flag) {
            case COLOR -> 3;
            case GRAYSCALE -> 1;
        };

        ByteBuffer bb = manager.allocateDirect(channel * height * width);
        // get an array of integer pixels in the default RGB color mode
        int[] pixels = image.getRGB(0, 0, width, height, null, 0, width);
        for (int rgb : pixels) {
            int alpha = (rgb >> 24) & 0xFF;
            int red = (rgb >> 16) & 0xFF;
            int green = (rgb >> 8) & 0xFF;
            int blue = rgb & 0xFF;

            if (flag == Image.Flag.GRAYSCALE) {
                bb.put((byte) red);
            } else if (flag == Image.Flag.COLOR) {
                bb.put((byte) red);
                bb.put((byte) green);
                bb.put((byte) blue);
            } else {
                bb.put((byte) alpha);
                bb.put((byte) red);
                bb.put((byte) green);
                bb.put((byte) blue);
            }
        }
        bb.rewind();
        return manager.create(bb, new Shape(height, width, channel), DataType.UINT8);
    }

    @Override
    public long availableSize() {
        return imageDataEntries.size();
    }

    private @NotNull BufferedImage readBufferedImage(String row, String col, boolean resize) {
        var key = row + "@" + col + "@" + resize;
        return copyImage(IMAGE_CACHE.computeIfAbsent(key, (i) -> {
            try (var imageInputStream = DataSetMain.class.getResourceAsStream(Objects.requireNonNull(imageTable.get(row, col)))) {
                Objects.requireNonNull(imageInputStream);
                var tmpImage = ImageIO.read(imageInputStream);
                if (resize) {
                    return resize32Image(tmpImage);
                } else {
                    return tmpImage;
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }));
    }

    public static @NotNull BufferedImage resize32Image(@NotNull BufferedImage inputImage) {
        var image = new BufferedImage(32, 32, inputImage.getType());
        var g = image.createGraphics();
        g.drawImage(inputImage, 0, 0, 32, 32, null);
        g.dispose();
        return image;
    }

    private @NotNull BufferedImage copyImage(@NotNull BufferedImage another) {
        var copy = new BufferedImage(another.getWidth(), another.getHeight(), another.getType());
        var g2d = copy.createGraphics();
        g2d.drawImage(another, 0, 0, null);
        g2d.dispose();
        return copy;
    }

    @Override
    public void prepare(Progress progress) throws IOException {
        if (!imageDataEntries.isEmpty()) {
            return;
        }
        imageTable.rowKeySet().parallelStream().forEach(row -> {
            imageTable.columnKeySet().parallelStream().forEach(col -> {
                if (imageTable.get(row, col) == null) {
                    return;
                }
                BufferedImage image;
                BufferedImage label;
                for (int i = 0; i < ((length - 1) / imagePathEntries.size() + 1); i++) {
                    // Adjust the diversity of data set by rotation here
                    var rotateDegree = ThreadLocalRandom.current().nextInt(12) * 30;
                    var mirrorLR = ThreadLocalRandom.current().nextBoolean();
                    var mirrorUD = ThreadLocalRandom.current().nextBoolean();
                    image = ImageUtils.rotate(readBufferedImage(row, col, true), rotateDegree);
                    if (mirrorLR) {
                        image = ImageUtils.mirrorLR(image);
                    }
                    if (mirrorUD) {
                        image = ImageUtils.mirrorUD(image);
                    }
                    // select random col
                    String materialCol;
                    {
                        var arr = new ArrayList<>(imageTable.columnKeySet());
                        do {
                            var index = ThreadLocalRandom.current().nextInt(arr.size());
                            materialCol = arr.remove(index);
                            if (imageTable.get(row, materialCol) == null) {
                                materialCol = null;
                            }
                        } while (materialCol == null);
                    }
                    label = ImageUtils.rotate(readBufferedImage(row, materialCol, false), rotateDegree);
                    if (mirrorLR) {
                        label = ImageUtils.mirrorLR(label);
                    }
                    if (mirrorUD) {
                        label = ImageUtils.mirrorUD(label);
                    }
                    // process the image
                    process(image, label);
                    imageDataEntries.add(new ImagePair(image, label));
                    if (imageDataEntries.size() >= length) {
                        return;
                    }
                }
            });
        });
    }

    public static @Nullable List<Integer> getPalette(@NotNull BufferedImage image) {
        var paletteArray = ColorThief.getPalette(image, 8, 5, true);
        var palette = new ArrayList<Integer>();
        if (paletteArray != null) {
            // Tabling is the most intuitive way in the coding world
            byte[] repeatSize = switch (paletteArray.length) {
                case 1 -> new byte[]{8};
                case 2 -> new byte[]{4, 4};
                case 3 -> new byte[]{3, 3, 2};
                case 4 -> new byte[]{2, 2, 2, 2};
                case 5 -> new byte[]{2, 2, 2, 1, 1};
                case 6 -> new byte[]{2, 2, 1, 1, 1, 1};
                case 7 -> new byte[]{2, 1, 1, 1, 1, 1, 1};
                case 8 -> new byte[]{1, 1, 1, 1, 1, 1, 1, 1};
                default -> throw new IllegalStateException("Unexpected value: " + paletteArray.length);
            };
            for (int j = 0, paletteArrayLength = paletteArray.length; j < paletteArrayLength; j++) {
                int[] rgbArr = paletteArray[j];
                for (int k = 0; k < repeatSize[j]; k++) {
                    palette.add(new Color(rgbArr[0] - 1, rgbArr[1] - 1, rgbArr[2] - 1).getRGB());
                }
                if (palette.size() > 8) {
                    break;
                }
            }
        } else {
            return null;
        }
        return palette;
    }

    @Contract("!null, !null -> param1")
    public static BufferedImage process(BufferedImage image, BufferedImage colorImage) {
        // repaint the black background with white
        var white = Color.WHITE.getRGB();
        var black = Color.BLACK.getRGB();
        for (int x = 0; x < 32; x++) {
            for (int y = 0; y < 32; y++) {
                if (image.getRGB(x, y) == black) {
                    image.setRGB(x, y, white);
                }
            }
        }
        var palette = getPalette(colorImage);
        if (palette == null) {
            throw new IllegalArgumentException("Failed to get palette from colorImage.");
        }
        // mix color
        int imgRgb;
        int pitRgb;
        for (int x = 0; x < 32; x++) {
            var pit = palette.iterator();
            for (int y = 0; y < 32; y++) {
                imgRgb = image.getRGB(x, y);
                if (!pit.hasNext()) {
                    pit = palette.iterator();
                }
                pitRgb = pit.next();
                // mix imgRgb and pitRgb
                var imgRed = (imgRgb >> 16) & 0xFF;
                var imgGreen = (imgRgb >> 8) & 0xFF;
                var imgBlue = imgRgb & 0xFF;
                var pitRed = (pitRgb >> 16) & 0xFF;
                var pitGreen = (pitRgb >> 8) & 0xFF;
                var pitBlue = pitRgb & 0xFF;
                var red = (imgRed + pitRed) / 2;
                var green = (imgGreen + pitGreen) / 2;
                var blue = (imgBlue + pitBlue) / 2;
                image.setRGB(x, y, (red << 16) | (green << 8) | blue);
            }
        }
        return image;
    }

    record ImagePair(BufferedImage image, BufferedImage label) {
    }

    @Contract(value = " -> new", pure = true)
    public static @NotNull EchoDataSetBuilder builder() {
        return new EchoDataSetBuilder();
    }

    public static class EchoDataSetBuilder extends BaseBuilder<EchoDataSetBuilder> {
        int length = 512;

        @Override
        protected EchoDataSetBuilder self() {
            return this;
        }

        public EchoDataSetBuilder length(int length) {
            this.length = length;
            return this;
        }

        public EchoDataSet build() {
            return new EchoDataSet(this, length);
        }
    }
}
