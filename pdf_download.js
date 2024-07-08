import { chromium } from 'playwright'; // Importing Playwright's WebKit module
import fs from 'fs';
import path from 'path';


// Specify the directory where you want to save the PDF
const downloadPath = path.join('.', 'downloads');
fs.mkdirSync(downloadPath, { recursive: true });

// Launching the browser
const browser = await chromium.launch({ headless: false });
const context = await browser.newContext({
    // Setting the download path for the browser context
    downloadsPath: downloadPath
});

// Opening a new page
const page = await context.newPage();

// Intercepting PDF requests and modifying the headers to force download
// Note that we are using endsWith() function here for a more
// loose catch-all strategy for all pdf files for a
// more generic approach
await page.route('**/*', async (route, request) => {
    if (request.resourceType() === 'document' && route.request().url().endsWith('.pdf')) {
        const response = await page.context().request.get(request);
        await route.fulfill({
            response,
            headers: {
                ...response.headers(),
                'Content-Disposition': 'attachment',
            }
        });
    } else {
        route.continue();
    }
});

// Navigate to the page where the PDF can be downloaded
// await page.goto('https://google.com')
await page.goto('https://www.genco3.com/Data/Sites/1/media/bao-cao-thuong-nien/2021/20210422_pgv_ar2020.pdf')
// await page.goto('https://www.engie.com/sites/default/files/assets/documents/2024-02/FY%202023%20Presentation%20-%20VDEF_2.pdf')
// await page.goto('https://www.uniper.energy/sites/default/files/2023-11/2022-11_Uniper_Capital_Markets_Story.pdf');

// Additional steps to trigger the PDF download, if necessary
// For example: await page.click('selector-to-download-button');

// Wait for the download to complete
page.on('download', async (download) => {
    const downloadUrl = download.url();
    const filePath = await download.path();
    console.log(`Downloaded file from ${downloadUrl} to ${filePath}`);
});

// Keep the browser open for a short period to ensure download completes
await new Promise(resolve => setTimeout(resolve, 10000));

// Close the browser
await browser.close();
