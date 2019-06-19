from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, IntegerField, SubmitField, BooleanField, validators, ValidationError

def integerValidator(form, field):
    if type(field.data) is None or field.data < 0:
        raise ValidationError('Must be an integer larger than zero!')

def floatValidator(form, field):
    if type(field.data) is None:
        raise ValidationError('Must be a float!')

class dataForm(FlaskForm):
    age = IntegerField('Age in Years', [validators.DataRequired(), validators.NumberRange(min=0, message='Input an age that is zero or greater!')])
    gender = SelectField('Gender', [validators.DataRequired()], choices=[(0, 'Male'), (1, 'Female')])
    total_bilirubin = FloatField('Total Bilirubin in mg/dL', [validators.DataRequired(), floatValidator])
    direct_bilirubin = FloatField('Conjugated Bilirubin in mg/dL', [validators.DataRequired(), floatValidator])
    alkaline_phosphate = IntegerField('Alkaline Phosphate in IU/L', [validators.DataRequired(), floatValidator])
    alamine_aminotransferase = IntegerField('Alamine Aminotransferase in IU/L', [validators.DataRequired(), floatValidator])
    aspartate_aminotransferase = IntegerField('Aspartate Aminotransferase in IU/L', [validators.DataRequired(), floatValidator])
    total_proteins = FloatField('Total Proteins in g/dL', [validators.DataRequired(), floatValidator])
    albumin = FloatField('Albumin in g/dL', [validators.DataRequired(), floatValidator])
    albumin_and_globulin_ratio = FloatField('A/G Ratio', [validators.DataRequired(), floatValidator])
    accept_tos = SelectField('I accept the TOS', [validators.DataRequired()], choices=[(0, 'Yes'), (1, 'No')])
    submit = SubmitField('Get Results')
