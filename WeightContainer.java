public class WeightContainer {

    double[][] weightMatrixV;
    double[][] weightMatrixW;
    
    public WeightContainer(double[][] weightMatrixV,double[][] weightMatrixW){
        this.weightMatrixV = weightMatrixV;
        this.weightMatrixW = weightMatrixW;
    }
    
    public WeightContainer(){}
    
    public double[][] getWeightMatrixV(){
        return this.weightMatrixV;
    }
    public double[][] getWeightMatrixW(){
        return this.weightMatrixW;
    }
    
    public void setWeights(double[][] weightMatrixV,double[][] weightMatrixW){
        this.weightMatrixV = weightMatrixV;
        this.weightMatrixW = weightMatrixW;
    }
    
}
