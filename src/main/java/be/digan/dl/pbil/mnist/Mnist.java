package be.digan.dl.pbil.mnist;

import be.digan.dl.pbil.neuralnet.Experiment;
import be.digan.dl.pbil.neuralnet.NeuralNet;
import be.digan.dl.pbil.PbilNeuralNetTrainer;
import be.digan.dl.pbil.neuralnet.activation.Relu;
import be.digan.dl.pbil.neuralnet.activation.Tanh;
import be.digan.dl.pbil.neuralnet.layer.DenseLayer;
import be.digan.dl.pbil.neuralnet.layer.FlattenMin;
import be.digan.dl.pbil.neuralnet.layer.Softmax;
import be.digan.dl.pbil.pbil.TrainingLogger;
import be.digan.dl.pbil.pbil.TrainingParameters;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Date;

public class Mnist {
    private static final Logger LOG = Logger.getLogger(Mnist.class);
    private static final String MNIST_FOLDER = "/home/david/MNIST";
    private static Experiment[] mnist_test;
    private static Experiment[] mnist_data;

    public static void main(String[] args) throws IOException {
        load_data();
    }

    private static void load_data() throws IOException {
        mnist_test = read_csv(MNIST_FOLDER + "/mnist_test.csv");
        mnist_data = read_csv(MNIST_FOLDER + "/mnist_train.csv");
        TrainingParameters parameters = new TrainingParameters();
//        NeuralNet net = getTripleLayerExperiment();
        NeuralNet net = getDualLayerExperiment();
//        NeuralNet net = getSingleLayerExperiment();
        MnistValidator validator = new MnistValidator(mnist_data, net, parameters);
        TrainingLogger logger = new MnistLogger(Arrays.copyOf(mnist_test, 1000), net);
        PbilNeuralNetTrainer trainer
                = new PbilNeuralNetTrainer(parameters, net, validator, logger);

        long before = new Date().getTime();
        long[] genotype = trainer.trainNetwork();
        TrainingLogger finalLogger = new MnistLogger(mnist_test, net);
        LOG.info("winner: " + Arrays.toString(genotype));
        finalLogger.fullLog("FINAL RESULT", genotype);

        LOG.info("Spend " + ((new Date().getTime() - before)/1000) + " sec training");
    }

    private static NeuralNet getSingleLayerExperiment() {
        NeuralNet net = new NeuralNet();
        net.addLayer(new DenseLayer(28*28,10, new Relu()));
        net.addLayer(new Softmax());
        return net;
    }
    private static NeuralNet getDualLayerExperiment() {
        NeuralNet net = new NeuralNet();
        net.addLayer(new DenseLayer(28*28,800, new Tanh()));
        net.addLayer(new DenseLayer(800,10, new Tanh()));
        net.addLayer(new FlattenMin());
        return net;
    }
    private static NeuralNet getTripleLayerExperiment() {
        NeuralNet net = new NeuralNet();
        net.addLayer(new DenseLayer(28*28,100, new Tanh()));
        net.addLayer(new DenseLayer(100,50, new Tanh()));
        net.addLayer(new DenseLayer(50,10, new Tanh()));
//        net.addLayer(new Softmax());
        net.addLayer(new FlattenMin());
        return net;
    }

    private static Experiment[] read_csv(String file) throws IOException {
        return Files.lines(Paths.get(file)).map(s -> readline(s)).toArray(Experiment[]::new);
    }

    private static Experiment readline(String s) {
        String[] split = s.split(",");
        int output = new Integer(split[0]);
        long[] input = Arrays.stream(split, 1, split.length)
                .map(String::trim).mapToLong(Integer::parseInt).map(i -> i * NeuralNet.FACTOR).toArray();
        return new Experiment(input, output);
    }
}
