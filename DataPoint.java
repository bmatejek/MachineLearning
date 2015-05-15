package MachineLearning;

public class DataPoint {
  private int[] attributes;  // list of attributes
  private int label;         // class label of this data point
  private boolean isBinary;  // is this variable integer or binary values
  
  // constructor given a data point
  public DataPoint(DataPoint that) {
    this.attributes = new int[that.attributes.length];
    for (int i = 0; i < this.attributes.length; ++i)
      this.attributes[i] = that.attributes[i];
    
    this.label = that.label;
    this.isBinary = that.isBinary;
  }
    
  // constructor for the data point
  public DataPoint(int[] attributes, int label, boolean isBinary) {
    this.attributes = new int[attributes.length];
    for (int i = 0; i < this.attributes.length; ++i) 
      this.attributes[i] = attributes[i];
    
    this.label = label;
    this.isBinary = isBinary;
  }
  
  // return the number of attributes
  public int NAttributes() {
    return attributes.length;
  }
  
  // accessor for the kth attribute
  public int KthAttribute(int k) {
    if (k < 0 || k >= attributes.length) {
      System.err.printf("Error: accessed illegal attribute %d.\n", k);
      return 0;
    }
    return attributes[k];
  }
  
  public int[] Attributes() {
    int[] ret = new int[attributes.length];
    for (int i = 0; i < attributes.length; ++i) 
      ret[i] = attributes[i];
    return ret;
  }
  
  // accessor for the label of this data point
  public int Label() {
    return label;
  }
  
  // is this a binary data point
  public boolean isBinary() {
    return isBinary;
  }
  
  // print out this data point
  public String toString() {
    String string = "";
    for (int i = 0; i < attributes.length; ++i) {
      string += Integer.toString(attributes[i]) + "\t";
    }
    string += Integer.toString(label);
    return string;
  }
}