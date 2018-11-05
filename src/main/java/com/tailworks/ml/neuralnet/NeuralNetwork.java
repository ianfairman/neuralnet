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
    private List<Layer> layers = new ArrayList<>();
    private List<Matrix> deltaWeights = new ArrayList<>();
    private List<Vec> deltaBias = new ArrayList<>();
    private CostFunction costFunction;


    private NeuralNetwork(Builder nb) {
        learningRate = nb.learningRate;
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
        Layer lastLayer = getLastLayer();
        Layer layer = lastLayer;

        do {
            Layer precedingLayer = layer.getPrecedingLayer();
            Vec out = layer.getOut();

            Vec dE_dO = layer == lastLayer ?        // Change of Error w.r.p change of output
                    costFunction.getDerivative(wanted, out) :
                    layer.getBackpropError();  // error prepared in last layer for us

            Vec dO_dI = layer.getActivation().dFn(out);

            // prepare error propagation and store in
            Vec dE_dI = dE_dO.elementProduct(dO_dI);
            Vec backpropErrorToNextLayer = layer.getWeights().multiply(dE_dI);
            precedingLayer.setBackpropError(backpropErrorToNextLayer);

            Matrix dE_dW = dE_dI.outerProduct(precedingLayer.getOut());

            deltaWeights.add(0, dE_dW);
            deltaBias.add(0, dE_dI);

            layer = precedingLayer;
        }
        while (layer.hasPrecedingLayer());      // Stop when we are at input layer


        // ----------------------------------
        // Update weights
        // ----------------------------------
        int cnt = 0;
        for (Layer l : layers)
            if (notFirstLayer(l)) {
                l.getWeights().subtract(deltaWeights.get(cnt).scale(learningRate));
                Vec newBias = l.getBias().subtract(deltaBias.get(cnt).scale(learningRate));
                l.setBias(newBias);
                cnt++;
            }

        deltaWeights.clear();
        deltaBias.clear();
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
                                new JsonPrimitive((double) src.doubleValue())
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
        private double learningRate = 0.25;
        private Initializer initializer = new Initializer.Random(0.5, 2);
        private CostFunction costFunction = new CostFunction.Quadratic();

        public Builder(int networkInputSize) {
            this.networkInputSize = networkInputSize;
        }

        public Builder setLearningRate(double learningRate) {
            this.learningRate = learningRate;
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
            log.info("Created NeuralNetwork: " + network.toJson(false).substring(0, 100) + " ...");
            return network;
        }

    }

    // -----------------------------

    private static class NetworkState {
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
    }
}

