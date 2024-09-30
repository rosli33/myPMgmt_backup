from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()

    def clean_file(self):
        file = self.cleaned_data.get('file')

        if not file.name.endswith('.csv'):
            raise forms.ValidationError('Invalid file type. Please upload a CSV file.')

        return file
