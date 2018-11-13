package com.tailworks.ml.neuralnet;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final double learningRate;
    private final int batchSize;
    private final CostFunction costFunction;

    private List<Layer> layers = new ArrayList<>();

    private NeuralNetwork(Builder nb) {
        learningRate = nb.learningRate;
        batchSize = nb.batchSize;
        costFunction = nb.costFunction;

        // Adding inputLayer
        Layer inputLayer = new Layer(nb.networkInputSize, Activation.Identity);
        layers.add(inputLayer);

        Layer precedingLayer = inputLayer;

        for (int i = 0; i < nb.layers.size(); i++) {
            Layer layer = nb.layers.get(i);
            Matrix w = new Matrix(precedingLayer.size(), layer.size());
            nb.initializer.initWeights(w, i);
            layer.addWeights(w);
            layers.add(layer);
            layer.setPrecedingLayer(precedingLayer);
            precedingLayer = layer;
        }
    }


    public Result evaluate(Vec input) {
        return evaluate(input, null);
    }


    public Result evaluate(Vec input, Vec wanted) {
        Vec signal = input;
        for (Layer layer : layers)
            signal = layer.feedWith(signal).getOut();


        if (wanted != null) {
            double cost = costFunction.getTotal(wanted, signal);
            return new Result(signal, cost);
        }
        return new Result(signal);
    }


    public void learn(Vec wanted) {
        Vec dEdO = null;
        Layer layer = getLastLayer();

        do {
            Vec out = layer.getOut();
            Layer precedingLayer = layer.getPrecedingLayer();

            if (dEdO == null) // first round (i.e. output-layer)
                dEdO = costFunction.getDerivative(wanted, out);

            Vec dEdI = layer.getActivation().dEdI(out, dEdO);

            // prepare error propagation and store for next iteration
            dEdO = layer.getWeights().multiply(dEdI);

            Matrix dEdW = dEdI.outerProduct(precedingLayer.getOut());

            layer.addDeltaWeights(dEdW);
            layer.addDeltaBias(dEdI);

            layer = precedingLayer;
        }
        while (layer.hasPrecedingLayer());      // Stop when we are at input layer


        // ----------------------------------
        // Update weights
        // ----------------------------------
        for (Layer l : layers) {
            if (notFirstLayer(l)) {
                l.updateWeights(learningRate);
                l.updateBias(learningRate);
            }
        }
    }

    private boolean notFirstLayer(Layer layer) {
        return layer.getPrecedingLayer() != null;
    }


    private Layer getLastLayer() {
        return layers.get(layers.size() - 1);
    }

    public String toJson(boolean pretty) {
        GsonBuilder gsonBuilder = new GsonBuilder()
                .registerTypeAdapter(Double.class,
                        (JsonSerializer<Double>) (src, typeOfSrc, context) ->
                                new JsonPrimitive(src.floatValue())
                );
        if (pretty) gsonBuilder.setPrettyPrinting();
        return gsonBuilder.create().toJson(new NetworkState(this));
    }


    // --------------------------------------------------------------------

    /**
     * Simple builder for a NeuralNetwork
     */
    public static class Builder {
        private static Logger log = LoggerFactory.getLogger(Builder.class);

        private List<Layer> layers = new ArrayList<>();
        private int networkInputSize;

        // defaults:
        private double learningRate = 0.005;
        private Initializer initializer = new Initializer.Random(-0.5, 0.5);
        private CostFunction costFunction = new CostFunction.MSE();
        private int batchSize = 1;

        public Builder(int networkInputSize) {
            this.networkInputSize = networkInputSize;
        }

        public Builder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder initWeights(Initializer initializer) {
            this.initializer = initializer;
            return this;
        }

        public Builder setCostFunction(CostFunction costFunction) {
            this.costFunction = costFunction;
            return this;
        }

        public Builder addLayer(Layer layer) {
            layers.add(layer);
            return this;
        }

        public NeuralNetwork create() {
            NeuralNetwork network = new NeuralNetwork(this);
            log.info("Created NeuralNetwork: " + network.toJson(false).substring(0, 200) + " ...");
            return network;
        }

    }

    // -----------------------------

    public static class NetworkState {
        double learningRate;
        String costFunction;
        Layer.LayerState[] layers;

        public NetworkState(NeuralNetwork network) {
            learningRate = network.learningRate;
            costFunction = network.costFunction.getName();

            layers = new Layer.LayerState[network.layers.size()];
            for (int l = 0; l < network.layers.size(); l++) {
                layers[l] = network.layers.get(l).getState();
            }
        }

        public Layer.LayerState[] getLayers() {
            return layers;
        }
    }
}

