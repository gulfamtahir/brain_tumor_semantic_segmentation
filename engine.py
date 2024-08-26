#Training Loop of the Model
# 7. Training Loop
num_epochs = 4
for epoch in range(num_epochs):
    compile_model.train()
    train_loss = 0.0
    for images, masks in train_data_loader:
        
        optimizer.zero_grad()
        outputs = compile_model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_data_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')
    
    # Validation Loop
    compile_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = compile_model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_data_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
