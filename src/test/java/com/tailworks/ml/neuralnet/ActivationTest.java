package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ActivationTest {

    @Test
    public void testSoftMax() {
        Activation softmax = Activation.Softmax;
        Vec v = softmax.fn(new Vec(-1, 0, 1.5, 2));
        assertEquals(v, new Vec(0.02778834297343303, 0.0755365477476706, 0.3385313204518047, 0.5581437888270917));
    }

    @Test
    public void testLReLU() {
        Activation softmax = Activation.Leaky_ReLU;
        Vec v = softmax.fn(new Vec(-1, 0, 1.5, 2));
        assertEquals(v, new Vec(-0.01, 0, 1.5, 2));
    }

}