public class TestFieldsLifter {
    // Static fields of different types
    private static int staticIntField = 42;
    public static String staticStringField = "Hello World";
    protected static double staticDoubleField = 3.14159;
    
    // Instance fields of different types
    private int instanceIntField;
    public String instanceStringField;
    protected double instanceDoubleField;
    
    // Constructor to initialize instance fields
    public TestFieldsLifter() {
        this.instanceIntField = 100;
        this.instanceStringField = "Instance Hello";
        this.instanceDoubleField = 2.71828;
    }
    
    public static void main(String[] args) {
        // Access and modify static fields
        System.out.println("Initial static fields:");
        System.out.println("staticIntField: " + staticIntField);
        System.out.println("staticStringField: " + staticStringField);
        System.out.println("staticDoubleField: " + staticDoubleField);
        
        // Modify static fields
        staticIntField = 84;
        staticStringField = "Modified Hello";
        staticDoubleField = 6.28318;
        
        System.out.println("\nModified static fields:");
        System.out.println("staticIntField: " + staticIntField);
        System.out.println("staticStringField: " + staticStringField);
        System.out.println("staticDoubleField: " + staticDoubleField);
        
        // Create instance and access instance fields
        TestFieldsLifter instance = new TestFieldsLifter();
        
        System.out.println("\nInitial instance fields:");
        System.out.println("instanceIntField: " + instance.instanceIntField);
        System.out.println("instanceStringField: " + instance.instanceStringField);
        System.out.println("instanceDoubleField: " + instance.instanceDoubleField);
        
        // Modify instance fields
        instance.instanceIntField = 200;
        instance.instanceStringField = "Modified Instance Hello";
        instance.instanceDoubleField = 1.41421;
        
        System.out.println("\nModified instance fields:");
        System.out.println("instanceIntField: " + instance.instanceIntField);
        System.out.println("instanceStringField: " + instance.instanceStringField);
        System.out.println("instanceDoubleField: " + instance.instanceDoubleField);
    }
}