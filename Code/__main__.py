def createFolders():
    OPEN_FOLDER = "../Data/Raw_Data/"
    SAVE_FOLDER = "../Data/Temp_Data/"
    IMAGE_FOLDER = "../Images/"

    open_1 = OPEN_FOLDER + "Messdaten_Test_ID_1/"
    open_4b = OPEN_FOLDER + "Messdaten_Test_ID_4b/"
    open_9 = OPEN_FOLDER + "Messdaten_Test_ID_9/"

    pp.checkFolder(SAVE_FOLDER)
    pp.checkFolder(OPEN_FOLDER)
    pp.checkFolder(IMAGE_FOLDER)
    pp.checkFolder(open_1)
    pp.checkFolder(open_4b)
    pp.checkFolder(open_9)