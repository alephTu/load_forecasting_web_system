// Form validation 

// Get current and past date
var date = new Date();
var pastDate = new Date();

var pastYear = date.getFullYear() - 2;
pastDate.setFullYear(pastYear);

// Fill forms if necessary
var defaultValues = {
    city_code: 'C1',
    date: '2018-01-10',
    method :'FC_DNN',
    from_date: '2018-01-01',
    to_date: '2018-01-31',
}

// Proper formatting
function formatDate(date){  
    var mnth = String(date.getMonth() + 1);
    var dy = String(date.getDate());
    var fy = date.getFullYear();
    var year = `${fy.toString().split('')[2]}${fy.toString().split('')[3]}`
    var dateStr = `${mnth}/${dy}/${year}`;
    return(dateStr);
}

// Fill in the blanks
document.querySelectorAll('input.form-control').forEach((input)=>{
    var id = input.id;
    if(input.value == ''){
        input.value = defaultValues[id];
    }
});


// Validation on dates
function validDate(){
    var fromDate = Date.parse(document.querySelector('#from_date').value);
    var toDate = Date.parse(document.querySelector('#to_date').value);
    var formatting = [document.querySelector('input#from_date'),document.querySelector('input#to_date')].filter(d=> d.value.includes('/')).length == 2;
    return fromDate <= toDate && formatting;
}

// Can't request future data
function dateCap(){
    return Date.parse(document.querySelector('#to_date').value) <= Date.parse(formatDate(date));
}

// No missing data
function filledInputs(){
missingInputs = 0;
document.querySelectorAll('.form-control').forEach((input)=>{
    if(input.value == '' || input.value == ' '){
        missingInputs = 1;
        return;
    }
})
return missingInputs == 0;
}


// Form validation
document.querySelector('button.button').addEventListener('click',(e)=>{
    e.preventDefault();
    if(1==1){
        // Set for dynamic URL (ONLY ON FORECAST PAGE)
        if(document.querySelector('div.analysis')){
            var inputTicker = document.querySelector('input#city_code').value;
            document.querySelector('form.data-form').setAttribute('action',`/prediction/${inputTicker}`);
        }

        document.querySelector('form.data-form').submit();
    }

});
