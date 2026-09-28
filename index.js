document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultBox = document.getElementById('result-box');
    const resTemp = document.getElementById('res-temp');
    const resWind = document.getElementById('res-wind');
    const resPress = document.getElementById('res-press');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            month: document.getElementById('month').value,
            prev_temp: document.getElementById('prev_temp').value,
            prev_wind: document.getElementById('prev_wind').value,
            prev_press: document.getElementById('prev_press').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                resTemp.textContent = data.predicted_temperature;
                resWind.textContent = data.predicted_wind_speed;
                resPress.textContent = data.predicted_pressure;
                resultBox.classList.remove('hidden');
            } else {
                alert('حدث خطأ أثناء التنبؤ: ' + data.error);
            }
        } catch (err) {
            alert('تعذر الاتصال بالسيرفر، يرجى المحاولة لاحقاً.');
        }
    });
});