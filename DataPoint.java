public class DataPoint {
  // list of attributes
  private Attribute attributes[];
  // class label of this data point
  private Label label; 
  
  // constructor for the data point
  public DataPoint(Attribute attributes[], Label label) {
    this.attributes = new Attribute[attributes.length];
    for (int i = 0; i < this.attributes.length; ++i) 
      this.attributes[i] = attributes[i];
    this.label = label;
  }
}