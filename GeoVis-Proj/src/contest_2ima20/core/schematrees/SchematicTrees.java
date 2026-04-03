/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package contest_2ima20.core.schematrees;

import contest_2ima20.core.problem.ProblemDefinition;

/**
 *
 * @author wmeulema
 */
public class SchematicTrees extends ProblemDefinition<Input,Output> {

    @Override
    public Input createInputInstance() {
        return new Input();
    }

    @Override
    public Output createOutputFor(Input input) {
        return new Output(input);
    }
    
}
