package com.tailworks.ml.neuralnet.optimizer;


import com.tailworks.ml.neuralnet.math.Matrix;
import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class MomentumTest {

    public static final double EPS = 0.0000001;

    @Test
    public void testMomentumWeightUpdate() {
        Matrix W = new Matrix(new double[][]{{2, 3, 4}, {3, 4, 5}});
        Matrix dW = new Matrix(new double[][]{{.2, .3, .4}, {.3, .4, .5}});
        Optimizer o = new Momentum(0.05);
        o.updateWeights(W, dW);
        assertArrayEquals(new double[]{1.990, 2.985, 3.980}, W.getData()[0], EPS);
        assertArrayEquals(new double[]{2.985, 3.980, 4.975}, W.getData()[1], EPS);
        o.updateWeights(W, dW);
        assertArrayEquals(new double[]{1.9710, 2.9565, 3.9420}, W.getData()[0], EPS);
        assertArrayEquals(new double[]{2.9565, 3.9420, 4.9275}, W.getData()[1], EPS);
    }

    @Test
    public void testMomentumBiasUpdate() {
        Vec bias = new Vec(2, 3, 4);
        Vec db = new Vec(.2, .3, .4);
        Optimizer o = new Momentum(0.05);
        bias = o.updateBias(bias, db);
        assertArrayEquals(new double[]{1.990, 2.985, 3.980}, bias.getData(), EPS);
        bias = o.updateBias(bias, db);
        assertArrayEquals(new double[]{1.9710, 2.9565, 3.9420}, bias.getData(), EPS);
    }

}
