package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CostFunctionTest {

    @Test
    public void testCrossEntropyCost() {
        /*
        double cost = new  CrossEntropy(new Vec(1, 0, 0), new Vec(0.2698, 0.3223, 0.4078)).getTotalCost();
        assertEquals(0.965, cost, 0.01);
        */
    }

    @Test
    public void testQuadraticCost() {
        Vec wanted = new Vec(1, 2, 3);
        Vec actual = new Vec(4, -3, 7);
        CostFunction.Quadratic costFn = new CostFunction.Quadratic();
        double cost = costFn.getTotal(wanted, actual);
        assertEquals((3 * 3 + 5 * 5 + 4 * 4) / 2.0, cost, 0.01);

        Vec err = costFn.getDerivative(wanted, actual);
        assertEquals(new Vec(3, -5, 4), err);
    }

    @Test
    public void testQuadraticCost2() {
        Vec wanted = new Vec(0.01, 0.99);
        Vec actual = new Vec(0.75136507, 0.77292846);
        CostFunction.Quadratic costFn = new CostFunction.Quadratic();
        double cost = costFn.getTotal(wanted, actual);
        assertEquals(0.298371109, cost, 0.01);


    }

}
