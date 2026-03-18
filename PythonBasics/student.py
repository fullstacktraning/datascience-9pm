from person import Person
class Student(Person):
    def __init__(self,name,age,roll_no,marks):
        super().__init__(name,age)
        self.roll_no = roll_no
        self.marks = marks
    def show_student_details(self):
        self.show_person_details()
        print("Roll No :",self.roll_no)
        print("Marks :",self.marks)