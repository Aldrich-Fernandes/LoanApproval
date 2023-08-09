import pandas as pd
class Session:
    def __init__(self):
        self.Subject = self.getSubjectChoice()
        self.Level = self.getLevelChoice()

    def getSubjectChoice(self):
        SubjectChoice = int(input("1. Computer Science \nEnter your subject choice: "))
        while SubjectChoice != 1:
            SubjectChoice = int(input("1. Computer Science \nEnter your choice: "))
        return SubjectChoice
    
    def getLevelChoice(self):
        pass ## Similar to getSubjectChoice

    # Note Processing
    def EnterNotes(self):
        Notes = str(input("Enter your notes on the subject and level that you have chosen: "))
        self.NotesAnalysis(Notes)

    def NotesAnalysis(self, Notes):
        NotesSplit = Notes.split()
        QuestionsGenerate = []
        if (self.LevelChoice == 1) and (self.SubjectChoice == 1):
            GCSEComputerScienceFile = pd.read_csv("GCSEComputerScience.csv")
        for i in NotesSplit:
            QuestionsGenerate.append(GCSEComputerScienceFile[GCSEComputerScienceFile["Question"].str.contains(i)])
            print(QuestionsGenerate)

    # Test User

    # Methods that implement testing

def main():
    Session = Session()
    Choice = int(input("Do you want to \n 1). Enter Your own notes \n 2). Use preloaded notes"))
    if Choice == 1:
        Session.EnterNotes()
    else:
        raise NotImplemented # other options 
    