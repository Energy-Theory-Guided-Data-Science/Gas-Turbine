def createFolders():
    OPEN_FOLDER = "../Data/Raw_Data/"
    SAVE_FOLDER = "../Data/Temp_Data/"
    IMAGE_FOLDER = "../Images/"

    open_1 = OPEN_FOLDER + "Messdaten_Test_ID_1/"
    open_4b = OPEN_FOLDER + "Messdaten_Test_ID_4b/"
    open_9 = OPEN_FOLDER + "Messdaten_Test_ID_9/"

    gf.check_folder(SAVE_FOLDER)
    gf.check_folder(OPEN_FOLDER)
    gf.check_folder(IMAGE_FOLDER)
    gf.check_folder(open_1)
    gf.check_folder(open_4b)
    gf.check_folder(open_9)