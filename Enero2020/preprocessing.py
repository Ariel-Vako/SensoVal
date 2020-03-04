from sklearn.preprocessing import StandardScaler


def estandarizacion(train, test):
    std = StandardScaler()
    train = std.fit_transform(train)
    test = std.transform(test)
    return train, test
