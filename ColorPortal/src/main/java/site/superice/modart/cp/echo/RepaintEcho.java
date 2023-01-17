package texture.echo;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import org.jetbrains.annotations.NotNull;
import site.superice.modart.texture.data.EchoDataSet;

import javax.annotation.WillNotClose;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static site.superice.modart.texture.data.EchoDataSet.toNDArray;

public class RepaintEcho {
    public static void main(String[] args) {
        var modelPath = Paths.get("./TextureArt/target/model/echo");
        var inputImagePath = Paths.get("./DataSet/src/main/resources/color/gt5-plate/Amethyst Plate.png");
        var colorImagePath = Paths.get("./DataSet/src/main/resources/color/gt5-rod/Blue Steel Rod.png");

        try (var model = Model.newInstance("mlp")) {
            model.setBlock(TrainEcho.buildBlocks());
            model.load(modelPath);

            var outputPath = Path.of("./TextureArt/target/output/echo");
            //noinspection ResultOfMethodCallIgnored
            outputPath.toFile().mkdirs();
            try (var inputStream = new FileInputStream(inputImagePath.toFile());
                 var colorImageStream = new FileInputStream(colorImagePath.toFile())) {
                var inputImage = (BufferedImage) BufferedImageFactory.getInstance().fromInputStream(inputStream).getWrappedImage();
                // copy inputImage to tmpImage
                var colorImage = (BufferedImage) BufferedImageFactory.getInstance().fromInputStream(colorImageStream).getWrappedImage();
                ImageIO.write(repaint(model, inputImage, colorImage),
                        "png", outputPath.resolve(inputImagePath.getFileName().toString()).toFile());
            }
        } catch (TranslateException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static @NotNull BufferedImage repaint(@WillNotClose @NotNull Model model, BufferedImage shapeImage,
                                                 BufferedImage colorImage) throws TranslateException {
        var predictor = model.newPredictor(new NoopTranslator());
        var tmpImage = EchoDataSet.resize32Image(shapeImage);
        var processedInputImage = EchoDataSet.process(tmpImage, colorImage);
        var inputData = new NDList(NDImageUtils.toTensor(toNDArray(model.getNDManager(), processedInputImage, Image.Flag.COLOR)));
        var result = predictor.predict(inputData);
        BufferedImage predictImage;
        try (var predictData = result.get(0);
             var predictSlice = predictData.get("0, :, :, :");
             var predictImageArray = predictSlice.mul(256)) {
            predictImage = (BufferedImage) BufferedImageFactory.getInstance().fromNDArray(predictImageArray.toType(DataType.UINT8, false)).getWrappedImage();
        }
        var black = Color.BLACK.getRGB();
        for (int x = 0; x < predictImage.getWidth(); x++) {
            for (int y = 0; y < predictImage.getHeight(); y++) {
                var rgb = shapeImage.getRGB(x, y);
                if (rgb == black || rgb == 0) {
                    predictImage.setRGB(x, y, black);
                }
            }
        }
        return predictImage;
    }
}
