public class AttributeClass {
  // name of the attribute
  private String name;
  // id of this attribute
  private int id; 
  
  // constructor for the attribute class
  public AttributeClass(String name, int id) {
    this.name = name;
    this.id = id;
  }
  
  // return the name of the attribute
  public String Name() {
    return name;
  }
  
  // return the id of the attribute
  public int ID() {
    return id;
  }
}