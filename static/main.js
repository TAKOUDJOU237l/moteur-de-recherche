const dropZone   = document.getElementById("dropZone");
const fileInput  = document.getElementById("fileInput");
const hiddenFile = document.getElementById("hiddenFile");
const previewZone= document.getElementById("previewZone");
const previewImg = document.getElementById("previewImg");
const previewName= document.getElementById("previewName");
const searchBtn  = document.getElementById("searchBtn");

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;

  // Prévisualisation
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewZone.style.display = "block";
    searchBtn.disabled = false;
  };
  reader.readAsDataURL(file);

  // Transfert vers le formulaire
  const dt = new DataTransfer();
  dt.items.add(file);
  hiddenFile.files = dt.files;
}

// Clic sur la zone
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => handleFile(e.target.files[0]));

// Drag & Drop
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  handleFile(e.dataTransfer.files[0]);
});