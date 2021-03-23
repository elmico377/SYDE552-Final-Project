import matplotlib.pyplot as plt

def plot_model(model,current_title):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'], c='k') 
    plt.ylabel(current_title + ' training accuracy')
    plt.xlabel('epochs')   
    plt.twinx()
    plt.plot(model.history.history['loss'], c='b')
    plt.ylabel('training loss (error)')
    plt.title('training')

    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['val_accuracy'], c='k')
    plt.ylabel(current_title + ' testing accuracy')
    plt.xlabel('epochs')
    plt.twinx()
    plt.plot(model.history.history['val_loss'], c='b')
    plt.ylabel('testing loss (error)')
    plt.title('testing')
    plt.tight_layout()
    plt.show()

def plot_models(models):
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    for model in models:
        plt.plot(model.history.history['accuracy'], c='k') 
    plt.ylabel('training accuracy')
    plt.xlabel('epochs')   
    plt.twinx()
    for model in models:
        plt.plot(model.history.history['loss'], c='b')
        plt.ylabel('training loss (error)')
    plt.title('training')

    plt.subplot(1, 2, 2)
    for model in models:
        plt.plot(model.history.history['val_accuracy'], c='k')
    plt.ylabel('testing accuracy')
    plt.xlabel('epochs')
    plt.twinx()
    for model in models:
        plt.plot(model.history.history['val_loss'], c='b')
    plt.ylabel('testing loss (error)')
    plt.title('testing')
    plt.tight_layout()
    plt.show()