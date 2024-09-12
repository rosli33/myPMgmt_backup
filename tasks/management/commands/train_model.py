from django.core.management.base import BaseCommand
from tasks.model.train_model import train_model  # Import the function we just defined

class Command(BaseCommand):
    help = 'Train the model and save it as a pickle file.'

    def handle(self, *args, **kwargs):
        try:
            train_model()  # Call the function to train the model
            self.stdout.write(self.style.SUCCESS('Model trained and saved successfully!'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error training the model: {str(e)}'))
