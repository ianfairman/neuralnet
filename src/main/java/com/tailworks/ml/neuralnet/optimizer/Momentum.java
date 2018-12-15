package com.tailworks.ml.neuralnet.optimizer;

import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

// Todo implement
public class Momentum implements Optimizer{

    private double learningRate;
    private double echoRate;
    private Matrix lastDW;
    private Matrix lastDBias;

    public Momentum(double learningRate, double echoRate) {
        this.learningRate = learningRate;
        this.echoRate = echoRate;
    }

    @Override
    public void updateWeights(Matrix weights, Matrix dCdW) {
        throw new NotImplementedException();
    }

    @Override
    public Vec updateBias(Vec bias, Vec dCdB) {
        throw new NotImplementedException();
    }

    @Override
    public Optimizer copy() {
        return new Momentum(learningRate, echoRate);
    }
}
