package site.superice.modart.cp.echo;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import org.jetbrains.annotations.NotNull;
import site.superice.modart.cp.data.EchoDataSet;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

public class DrawEcho {
    public static void main(String[] args) {
        var modelPath = Paths.get("./ColorPortal/src/main/resources/model/base");

        var ds = EchoDataSet.builder().setSampling(64, true).build();
        try {
            ds.prepare(null);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try (var model = Model.newInstance("cp")) {
            model.setBlock(TrainEcho.buildBlocks());
            model.load(modelPath);
            var outputPath = Path.of("./ColorPortal/target/output/image");
            //noinspection ResultOfMethodCallIgnored
            outputPath.toFile().mkdirs();
            var result = randomDraw(ds, model, 10);
            int i = 0;
            for (var each : result.entrySet()) {
                ++i;
                ImageIO.write(each.getKey(), "png", new FileOutputStream(outputPath.resolve("debug-nd-expect-" + i + ".png").toFile()));
                ImageIO.write(each.getValue(), "png", new FileOutputStream(outputPath.resolve("debug-nd-result-" + i + ".png").toFile()));
            }
        } catch (MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return Map: ExpectImage, PredictImage
     */
    @NotNull
    public static Map<BufferedImage, BufferedImage> randomDraw(@NotNull EchoDataSet dataSet, @NotNull Model model, int count) {
        var predictor = model.newPredictor(new NoopTranslator());
        var outputMap = new HashMap<BufferedImage, BufferedImage>();
        for (int i = 0; i < count; i++) {
            try (var manager = NDManager.newBaseManager()) {
                var record = dataSet.get(manager, ThreadLocalRandom.current().nextInt((int) dataSet.availableSize()));
                NDList result;
                try {
                    result = predictor.predict(record.getData());
                } catch (TranslateException e) {
                    throw new RuntimeException(e);
                }
                var expectImage = BufferedImageFactory.getInstance().fromNDArray(record.getLabels().get(0).mul(256).toType(DataType.UINT8, false));
                var predictImage = BufferedImageFactory.getInstance().fromNDArray(result.get(0).get("0, :, :, :").mul(256).toType(DataType.UINT8, false));
                for (var x = 0; x < 16; x++) {
                    for (var y = 0; y < 16; y++) {
                        var color = ((BufferedImage) expectImage.getWrappedImage()).getRGB(x, y);
                        if (color == Color.BLACK.getRGB()) {
                            ((BufferedImage) predictImage.getWrappedImage()).setRGB(x, y, 0);
                        }
                    }
                }
                outputMap.put((BufferedImage) expectImage.getWrappedImage(), (BufferedImage) predictImage.getWrappedImage());
            }
        }
        return outputMap;
    }
}
