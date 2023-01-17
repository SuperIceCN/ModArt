package site.superice.modart.cp.data;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class EnrichmentDataSet extends RandomAccessDataset {
    protected final List<BufferedImage> imageList;

    public EnrichmentDataSet(Builder builder) {
        super(builder);
        this.imageList = builder.imageList;
    }

    @Override
    public Record get(NDManager manager, long index) {
        var valId = (int) index / imageList.size();
        var keyId = (int) index % imageList.size();
        var imageData = EchoDataSet.resize32Image(imageList.get(keyId));
        var labelData = imageList.get(valId);
        EchoDataSet.process(imageData, labelData);
        var dataTensor = NDImageUtils.toTensor(EchoDataSet.toNDArray(manager, imageData, Image.Flag.COLOR));
        dataTensor.setRequiresGradient(true);
        var dataNDList = new NDList(dataTensor);
        var labelNDList = new NDList(NDImageUtils.toTensor(EchoDataSet.toNDArray(manager, labelData, Image.Flag.COLOR)));
        return new Record(dataNDList, labelNDList);
    }

    @Override
    protected long availableSize() {
        return (long) imageList.size() * (imageList.size() - 1);
    }

    @Override
    public void prepare(Progress progress) {

    }

    @Contract(" -> new")
    public static @NotNull Builder builder() {
        return new Builder();
    }

    public static class Builder extends BaseBuilder<Builder> {
        protected List<BufferedImage> imageList = new ArrayList<>();

        @Override
        protected Builder self() {
            return this;
        }

        public EnrichmentDataSet build() {
            return new EnrichmentDataSet(this);
        }

        public Builder addImage(BufferedImage image) {
            imageList.add(image);
            return this;
        }

        public Builder addImages(List<BufferedImage> imageList) {
            this.imageList.addAll(imageList);
            return this;
        }
    }
}
