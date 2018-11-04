package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Matrix;
import org.junit.Test;

import static java.lang.Math.sqrt;
import static org.junit.Assert.assertEquals;

public class InitializerTest {

    @Test
    public void testHeUniform() {
        int in = 700;
        Matrix m = new Matrix(300, in);
        new Initializer.HeUniform().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(varOnUniformDist(-sqrt(6.0 / in), sqrt(6.0 / in)), m.variance(), 0.001);
    }

    @Test
    public void testHeNormal() {
        int in = 700;
        Matrix m = new Matrix(300, in);
        new Initializer.HeNormal().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(2.0 / in, m.variance(), 0.001);
    }

    @Test
    public void testXavierUniform() {
        int in = 700;
        int out = 300;
        Matrix m = new Matrix(out, in);
        new Initializer.XavierUniform().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(varOnUniformDist(-sqrt(6.0 / (in+out)), sqrt(6.0 / (in+out))), m.variance(), 0.001);
    }

    @Test
    public void testXavierNormal() {
        int in = 700;
        int out = 300;
        Matrix m = new Matrix(out, in);
        new Initializer.XavierNormal().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(2.0 / (in+out), m.variance(), 0.001);
    }


    @Test
    public void testLecunUniform() {
        int in = 700;
        int out = 300;
        Matrix m = new Matrix(out, in);
        new Initializer.LeCunUniform().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(varOnUniformDist(-sqrt(3.0 / in), sqrt(3.0 / in)), m.variance(), 0.0001);
    }

    @Test
    public void testLecunNormal() {
        int in = 700;
        int out = 300;
        Matrix m = new Matrix(out, in);
        new Initializer.LeCunNormal().initWeights(m, 0);
        assertEquals(0, m.average(), 0.001);
        assertEquals(2.0 / (in+out), m.variance(), 0.001);
    }



    private double varOnUniformDist(double a, double b) {
        return (b - a) * (b - a) / 12.0;
    }

}