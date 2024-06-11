function addImageInput() {
    var container = document.getElementById("imageInputs");
    var newInput = document.createElement("div");
    newInput.classList.add("mb-3");
    newInput.innerHTML = '<label for="inputGroupFile04" class="form-label">Input Gambar</label><input name="file" type="file" class="form-control imageInput" accept="image/*">';
    container.appendChild(newInput);
}