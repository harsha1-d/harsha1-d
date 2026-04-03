/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.core.problem;


/**
 *
 * @author Wouter Meulemans (w.meulemans@tue.nl)
 */
public abstract class ProblemDefinition<TInput extends Problem, TOutput extends Solution> {

    public abstract TInput createInputInstance();

    public abstract TOutput createOutputFor(TInput input);
}
