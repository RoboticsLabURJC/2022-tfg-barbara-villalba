
error_values = -30:30;

min_error = 0
max_error = max(error_values)
normalized_error = (error_values - min_error) / (max_error - min_error)
% 
% 
reward1 = 1 ./ (1 + abs(error_values));
reward2 = 1-abs(normalized_error);
reward3 = -exp(error_values);
reward4 = -log(error_values);
reward5 = 1./(1 + exp(-error_values));


% 
% % Crear una figura con dos subplots
figure;
% 
% % Subplot 1: Función de Recompensa 1
subplot(2, 3, 1);
plot(error_values, reward1);
xlabel('Error del Centro del Carril')
ylabel('Recompensa')
title('Función de Recompensa 1:Inversamente proporcional')
grid on
% 
% % Subplot 2: Función de Recompensa 2
subplot(2, 3, 2);
plot(error_values, reward2);
xlabel('Error del Centro del Carril')
ylabel('Recompensa')
title('Función de Recompensa 2:Linealmente')
grid on

subplot(2, 3, 3);
plot(error_values, reward3);
xlabel('Error del Centro del Carril')
ylabel('Recompensa')
title('Función de Recompensa 3: Exponencial')
grid on

subplot(2, 3, 4);
plot(error_values, reward4);
xlabel('Error del Centro del Carril')
ylabel('Recompensa')
title('Función de Recompensa 4: Logaritmica')
grid on

subplot(2, 3, 5);
plot(error_values, reward5);
xlabel('Error del Centro del Carril')
ylabel('Recompensa')
title('Función de Recompensa 5: Sigmoide')
grid on
% 
% % Ajustar el espaciado entre subplots
sgtitle('Representación de Funciones de Recompensa')

