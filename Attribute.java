public class Attribute {
  // the attribute class for this attribute
  private AttributeClass attribute_class;
  // the real value of this attribute
  private double value;
  
  // constructor for the attribute
  public Attribute(AttributeClass attribute_class, double value) {
    this.attribute_class = attribute_class;
    this.value = value;
  }
  
  // return the class of this attribute
  public AttributeClass AttributeClass() {
    return attribute_class;
  }
  
  // return the value of this attribute
  public double Value() {
    return value;
  }
}